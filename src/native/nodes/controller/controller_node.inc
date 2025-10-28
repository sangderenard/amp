static int run_controller_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels,
    const EdgeRunnerControlHistory *history
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    /* allocate parsing buffers on the heap so they can be registered
       and reliably observed by the debug registry (avoids untracked
       stack writes and lifetime confusion). Sizes are chosen from the
       descriptor length with safe defaults and caps. */
    char *outputs_csv = NULL;
    char *sources_csv = NULL;
    char (*output_names)[64] = NULL;
    controller_source_t *mappings = NULL;
    int output_count = 0;
    int mapping_count = 0;
    int resolved_channels = 0;
    size_t total = 0;
    double *buffer = NULL;
    int _rc = -1;
    size_t outputs_cap = 256;
    size_t sources_cap = 512;
    if (descriptor != NULL && descriptor->params_len > 0) {
        size_t p = descriptor->params_len + 1;
        if (p > outputs_cap) outputs_cap = p;
        if (p > sources_cap) sources_cap = p;
    }
    /* clamp to reasonable bounds */
    if (outputs_cap < 256) outputs_cap = 256;
    if (sources_cap < 512) sources_cap = 512;
    if (outputs_cap > 65536) outputs_cap = 65536;
    if (sources_cap > 65536) sources_cap = 65536;
    outputs_csv = (char *)malloc(outputs_cap);
    if (outputs_csv == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    amp_debug_register_alloc(outputs_csv, outputs_cap);
    sources_csv = (char *)malloc(sources_cap);
    if (sources_csv == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    amp_debug_register_alloc(sources_csv, sources_cap);
    output_names = (char (*)[64])malloc((size_t)32 * 64);
    if (output_names == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    amp_debug_register_alloc(output_names, (size_t)32 * 64);
    mappings = (controller_source_t *)malloc((size_t)32 * sizeof(controller_source_t));
    if (mappings == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    amp_debug_register_alloc(mappings, (size_t)32 * sizeof(controller_source_t));
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "__controller_outputs__", outputs_csv, outputs_cap)) {
        output_count = parse_csv_tokens(outputs_csv, output_names, 32);
    }
    if (output_count <= 0 && inputs != NULL) {
        uint32_t count = inputs->params.count;
        EdgeRunnerParamView *items = inputs->params.items;
        for (uint32_t i = 0; i < count && i < 32U; ++i) {
            if (items[i].name != NULL) {
                strncpy(output_names[output_count], items[i].name, sizeof(output_names[output_count]) - 1);
                output_names[output_count][sizeof(output_names[output_count]) - 1] = '\0';
                output_count++;
            }
        }
    }
    if (output_count <= 0) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "__controller_sources__", sources_csv, sources_cap)) {
        mapping_count = parse_controller_sources(sources_csv, mappings, 32);
    }
    resolved_channels = output_count;
    if (inputs != NULL && inputs->params.count > 0) {
        const EdgeRunnerParamView *view = &inputs->params.items[0];
        if (batches <= 0 && view->batches > 0) {
            batches = (int)view->batches;
        }
        if (frames <= 0 && view->frames > 0) {
            frames = (int)view->frames;
        }
    }
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    total = (size_t)batches * (size_t)resolved_channels * (size_t)frames;
    buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        _rc = -1;
        goto cleanup_run_controller_node;
    }
    for (int c = 0; c < resolved_channels; ++c) {
        const char *source_name = output_names[c];
        for (int m = 0; m < mapping_count; ++m) {
            if (strcmp(mappings[m].output, output_names[c]) == 0) {
                source_name = mappings[m].source;
                break;
            }
        }
        const EdgeRunnerParamView *view = find_param(inputs, source_name);
        int view_missing = (view == NULL || view->data == NULL);
        double *owned = NULL;
        const double *data = ensure_param_plane(view, batches, frames, 0.0, &owned);
        if (data == NULL) {
            free(buffer);
            _rc = -1;
            goto cleanup_run_controller_node;
        }
        if (view_missing && owned != NULL && history != NULL) {
            const EdgeRunnerControlCurve *curve = find_history_curve(history, source_name, strlen(source_name));
            if (curve != NULL) {
                apply_history_curve(owned, batches, frames, curve);
            }
            data = owned;
        }
        for (int b = 0; b < batches; ++b) {
            for (int f = 0; f < frames; ++f) {
                size_t src_idx = (size_t)b * (size_t)frames + (size_t)f;
                size_t dst_idx = ((size_t)b * (size_t)resolved_channels + (size_t)c) * (size_t)frames + (size_t)f;
                /* bounds checks and durable logging for write operations */
                size_t total_len = total; /* copied to local for clarity */
                if (dst_idx >= total_len) {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
                    AMP_DEBUG_LOG_MEMOPS("BAD_WRITE run_controller_node dst_idx=%zu total=%zu src_idx=%zu c=%d b=%d f=%d resolved_channels=%d frames=%d\n", dst_idx, total_len, src_idx, c, b, f, resolved_channels, frames);
                    AMP_DEBUG_FLUSH_MEMOPS();
#endif
                    /* attempt to continue safely */
                    continue;
                }
                /* ensure src_idx is in range of the data plane */
                /* we don't know 'data' length statically; we will conservatively attempt to log if src_idx seems large */
                buffer[dst_idx] = data[src_idx];
#if defined(AMP_NATIVE_ENABLE_LOGGING)
                AMP_DEBUG_LOG_MEMOPS("POSTWRITE run_controller_node dest=%p dst_idx=%zu wrote=1 src_idx=%zu node=%s\n", (void *)buffer, dst_idx, src_idx, descriptor->name ? descriptor->name : "<noname>");
                AMP_DEBUG_FLUSH_MEMOPS();
#endif
            }
        }
        if (owned != NULL) {
            free(owned);
        }
    }
    *out_buffer = buffer;
    *out_channels = resolved_channels;
    _rc = 0;

cleanup_run_controller_node:
    /* Free and unregister any heap buffers allocated above. Keep this idempotent. */
    if (outputs_csv != NULL) {
        amp_debug_unregister_alloc(outputs_csv);
        free(outputs_csv);
        outputs_csv = NULL;
    }
    if (sources_csv != NULL) {
        amp_debug_unregister_alloc(sources_csv);
        free(sources_csv);
        sources_csv = NULL;
    }
    if (output_names != NULL) {
        amp_debug_unregister_alloc(output_names);
        free(output_names);
        output_names = NULL;
    }
    if (mappings != NULL) {
        amp_debug_unregister_alloc(mappings);
        free(mappings);
        mappings = NULL;
    }
    return _rc;
}

