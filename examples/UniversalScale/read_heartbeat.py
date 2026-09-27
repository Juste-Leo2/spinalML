"""Read DramSoCTop heartbeat frames (0xA5 + status + rstinfo) from UART.

Terminal-only debug reader: prints one line per frame.
status: bit0 = pll_locked, bit1 = init_done, bit2 = sys_rst,
  bits[7:3] = init step idx.
rstinfo: bit7 = sys tick (toggles iff sys_clk runs), bit6 = por_done,
  bit5 = sys reset request, bit4 = crg.reset, bit3 = sys2x reset
  (0 proves sys2x_i toggles), bit0 = reset_n pin.
"""
import sys
import time

import serial
import serial.tools.list_ports


def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise SystemExit("no serial ports found")
    # Prefer USB serial (Tang Primer 20K) when several ports exist.
    for p in ports:
        if "USB" in (p.description or "").upper():
            return p.device
    return ports[0].device


def main():
    args = sys.argv[1:]
    port = None
    secs = 6.0
    for a in args:
        if a[:3].upper() == "COM" or a.startswith("/dev/"):
            port = a
        else:
            secs = float(a)
    if port is None:
        ports = list(serial.tools.list_ports.comports())
        print("ports: " + ", ".join(f"{p.device} ({p.description})"
                                    for p in ports))
        port = pick_port()
    print(f"listening on {port} ({secs}s) ...")
    ser = serial.Serial(port, 115200, timeout=0.2)
    buf = bytearray()
    t0 = time.time()
    frames = 0
    while time.time() - t0 < secs:
        chunk = ser.read(64)
        if chunk:
            buf += chunk
        while True:
            i = buf.find(b"\xa5")
            if i < 0:
                buf = buf[-1:] if buf and buf[-1] != 0xA5 else buf
                break
            if len(buf) - i < 3:
                break
            status = buf[i + 1]
            rstinfo = buf[i + 2]
            locked = bool(status & 0x01)
            init = bool(status & 0x02)
            sys_rst = bool(status & 0x04)
            step = status >> 3
            tick = bool(rstinfo & 0x80)
            por = bool(rstinfo & 0x40)
            req = bool(rstinfo & 0x20)
            creset = bool(rstinfo & 0x10)
            s2x = bool(rstinfo & 0x08)
            rpin = bool(rstinfo & 0x01)
            frames += 1
            print(f"frame #{frames}: status=0x{status:02X} "
                  f"pll_locked={int(locked)} init_done={int(init)} "
                  f"sys_rst={int(sys_rst)} step={step} "
                  f"rst=0x{rstinfo:02X} "
                  f"tick={int(tick)} por={int(por)} req={int(req)} "
                  f"creset={int(creset)} s2x_rst={int(s2x)} "
                  f"reset_n={int(rpin)}")
            del buf[: i + 3]
    ser.close()
    if frames == 0:
        print("NO HEARTBEAT: line silent (check baud/wiring/bitstream).")


if __name__ == "__main__":
    main()
