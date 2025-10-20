# menu.py

# =========================
# Menu (short)
# =========================
class Menu:
    def __init__(self,state): self.state=state; self.active=False
    def toggle(self): self.active=not self.active
    def draw(self):
        if not self.active: return
        s=self.state
        print("\n================ MENU ================")
        print(f"Source: {s['source_type']} (N to toggle)")
        print(f"Wave: {s['waves'][s['wave_idx']]}")
        print(f"Base token: {s['base_token']}  Root MIDI: {s['root_midi']}")
        print(f"FREE variant: {s['free_variant']}")
        print(f"LFO: wave={s['mod_wave_types'][s['mod_wave_idx']]} rate={s['mod_rate_hz']:.3f}Hz depth={s['mod_depth']:.2f} route={s['mod_route']} use_input={s['mod_use_input']}")
        print("Hold bumper = momentary mode; double-tap = latch")
        print("======================================\n")