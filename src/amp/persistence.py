# persistence.py
import os, json, copy
from .state import MAPPINGS_FILE, MAX_UNDO

def load_mappings(state):
    if os.path.exists(MAPPINGS_FILE):
        try:
            with open(MAPPINGS_FILE,"r",encoding="utf-8") as f:
                data = json.load(f)
            if "keymap" in data: state["keymap"].update(data["keymap"])
            if "buttonmap" in data: state.setdefault("buttonmap",{}).update(data["buttonmap"])
            state["_mapping_undo_stack"] = data.get("_mapping_undo_stack", [])
        except Exception as e:
            print(f"Failed to load mappings: {e}")
    else:
        state["_mapping_undo_stack"] = []

def save_mappings(state):
    try:
        data = {
            "keymap": state.get("keymap",{}),
            "buttonmap": state.get("buttonmap",{}),
            "_mapping_undo_stack": state.get("_mapping_undo_stack",[])
        }
        tmp = MAPPINGS_FILE + ".tmp"
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=2)
        os.replace(tmp,MAPPINGS_FILE)
    except Exception as e:
        print(f"Failed to save mappings: {e}")

def push_undo_snapshot(state):
    stack = state.setdefault("_mapping_undo_stack",[])
    snap = {"keymap":copy.deepcopy(state.get("keymap",{})),
            "buttonmap":copy.deepcopy(state.get("buttonmap",{}))}
    stack.append(snap)
    if len(stack)>MAX_UNDO:
        del stack[0:len(stack)-MAX_UNDO]
    state["_mapping_undo_stack"] = stack
    save_mappings(state)

def undo_mappings(state):
    stack = state.get("_mapping_undo_stack",[])
    if not stack:
        print("No undo snapshots.")
        return
    last = stack.pop()
    state["keymap"]=last.get("keymap",{})
    state["buttonmap"]=last.get("buttonmap",{})
    state["_mapping_undo_stack"]=stack
    save_mappings(state)
