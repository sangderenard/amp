import traceback

import scripts.demo_kpn_kpn_native_correct as demo

try:
    rc = demo.main(["--duration", "0.1", "--block-size", "256", "--out-dir", "output/demo_test"])
    print("rc", rc)
except Exception as exc:
    traceback.print_exc()
