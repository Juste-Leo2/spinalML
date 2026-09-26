/* Gowin blackbox stubs for cells used by LiteDRAM's GW2DDRPHY that the
 * open-source Yosys Gowin library (cells_sim.v) does not model.
 *
 * Why this file exists: Yosys `hierarchy -check` (run by `synth_gowin`)
 * rejects instantiations of undefined modules, so the generated
 * `litedram_core.v` cannot even elaborate without these declarations.
 * Declared `(* blackbox *)`, Yosys passes the cells through untouched and
 * nextpnr-himbaechel places the real hardware cells (verified: PnR accepts
 * DLL / IODELAY / ELVDS_IOBUF on GW2A-18, failing only on missing pin
 * constraints as expected). On silicon these are REAL cells: the DLL
 * genuinely self-calibrates at power-up (see GW2DDRPHYInit), IODELAYs apply
 * their configured static delay, DQS stays truly differential. No
 * functional compromise, no LiteDRAM fork.
 *
 * Coupling: port/parameter lists must match what LiteDRAM 2026.08 emits
 * (see dram/out/litedram_core.v). Re-check after any LiteDRAM bump:
 * `grep -A9 "^DLL \|^IODELAY \|^ELVDS_IOBUF " dram/out/litedram_core.v`.
 * SPDX-License-Identifier: MIT
 */

(* blackbox *) module DLL (
	input  CLKIN,
	input  RESET,
	input  STOP,
	input  UPDNCNTL,
	output LOCK,
	output [7:0] STEP
);
	parameter SCAL_EN = "false";
endmodule

(* blackbox *) module IODELAY (
	input  DI,
	input  SDTAP,
	input  SETN,
	input  VALUE,
	input  DF,
	output DO
);
	parameter C_STATIC_DLY = 1'd0;
endmodule

(* blackbox *) module ELVDS_IOBUF (
	input  I,
	input  OEN,
	inout  IO,
	inout  IOB,
	output O
);
endmodule
