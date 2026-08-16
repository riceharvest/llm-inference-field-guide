# Ling-3.0-tiny 128K auto-research beam log
Started 2026-08-16T01:35:19.619946Z | primary=decode tok/s (geomean), constraint=peak VRAM GB

- `A_par1` promoted | decode=153.154 vram=7.83 prefill=691.29 ttft=39.23 @ 2026-08-16T01:36:06.300590Z
- `A_par2` promoted | decode=153.181 vram=7.84 prefill=404.778 ttft=36.08 @ 2026-08-16T01:36:53.220262Z
- `A_par3` promoted | decode=154.303 vram=7.59 prefill=402.306 ttft=36.17 @ 2026-08-16T01:37:39.626995Z
- `A_par4` promoted | decode=154.702 vram=7.55 prefill=408.675 ttft=35.67 @ 2026-08-16T01:38:25.848965Z
- `A_par6` promoted | decode=154.692 vram=7.61 prefill=413.577 ttft=35.22 @ 2026-08-16T01:39:12.171342Z
- `A_par8` promoted | decode=155.578 vram=7.59 prefill=411.883 ttft=35.36 @ 2026-08-16T01:39:58.112050Z
- `B_kvbf16` promoted | decode=155.661 vram=7.52 prefill=404.847 ttft=35.92 @ 2026-08-16T01:40:44.038746Z
- `B_kvq8_0` promoted | decode=155.659 vram=7.58 prefill=399.55 ttft=36.5 @ 2026-08-16T01:41:27.388330Z
- `B_kvq4_0` promoted | decode=161.752 vram=7.11 prefill=423.832 ttft=34.37 @ 2026-08-16T01:42:02.772369Z
- `C_ub256_b512` promoted | decode=156.234 vram=7.77 prefill=403.172 ttft=36.45 @ 2026-08-16T01:42:48.502519Z
- `C_ub512_b1024` promoted | decode=156.647 vram=7.83 prefill=412.38 ttft=35.4 @ 2026-08-16T01:43:34.151935Z
- `C_ub1024_b2048` promoted | decode=155.031 vram=7.81 prefill=402.189 ttft=36.26 @ 2026-08-16T01:44:20.297169Z
- `C_ub2048_b4096` promoted | decode=153.08 vram=7.25 prefill=399.797 ttft=36.51 @ 2026-08-16T01:45:07.046388Z
- `D_t8_tb8` promoted | decode=152.809 vram=6.96 prefill=406.398 ttft=35.84 @ 2026-08-16T01:45:53.763691Z
- `D_t14_tb16` promoted | decode=153.292 vram=6.94 prefill=406.96 ttft=35.94 @ 2026-08-16T01:46:40.378982Z
- `D_t16_tb16` promoted | decode=152.927 vram=6.96 prefill=397.871 ttft=36.78 @ 2026-08-16T01:47:27.106171Z
- `D_t20_tb20` promoted | decode=153.131 vram=6.92 prefill=396.283 ttft=36.72 @ 2026-08-16T01:48:13.827789Z
- `E_kvq4_par8` promoted | decode=153.376 vram=6.37 prefill=405.377 ttft=35.87 @ 2026-08-16T01:48:51.546371Z

## Final ranking (18 promoted)

1. `B_kvq4_0` decode=161.752 vram=7.11 prefill=423.832
2. `C_ub512_b1024` decode=156.647 vram=7.83 prefill=412.38
3. `C_ub256_b512` decode=156.234 vram=7.77 prefill=403.172
4. `B_kvbf16` decode=155.661 vram=7.52 prefill=404.847
5. `B_kvq8_0` decode=155.659 vram=7.58 prefill=399.55
6. `A_par8` decode=155.578 vram=7.59 prefill=411.883
7. `C_ub1024_b2048` decode=155.031 vram=7.81 prefill=402.189
8. `A_par4` decode=154.702 vram=7.55 prefill=408.675
9. `A_par6` decode=154.692 vram=7.61 prefill=413.577
10. `A_par3` decode=154.303 vram=7.59 prefill=402.306
11. `E_kvq4_par8` decode=153.376 vram=6.37 prefill=405.377
12. `D_t14_tb16` decode=153.292 vram=6.94 prefill=406.96
13. `A_par2` decode=153.181 vram=7.84 prefill=404.778
14. `A_par1` decode=153.154 vram=7.83 prefill=691.29
15. `D_t20_tb20` decode=153.131 vram=6.92 prefill=396.283
16. `C_ub2048_b4096` decode=153.08 vram=7.25 prefill=399.797
17. `D_t16_tb16` decode=152.927 vram=6.96 prefill=397.871
18. `D_t8_tb8` decode=152.809 vram=6.96 prefill=406.398
