// SPDX-License-Identifier: MIT
// Ported from Vireo 9366da5e14dcafe846fc41565c94f4f145eaf7cc. MLX kernel body.
uint i = thread_position_in_grid.x;
int B = dims[0], L = dims[1], Hq = dims[2], Hk = dims[3], D = dims[4], Bc = dims[5];
int nq = B * L * Hq * D;
int nk = B * L * Hk * D;
if (int(i) >= nq + nk) return;
bool isq = int(i) < nq;
int j = isq ? int(i) : int(i) - nq;
int H = isq ? Hq : Hk;
int d = j % D;
int l = (j / (D * H)) % L;
int b = j / (D * H * L);
int ci = ((Bc == 1 ? 0 : b) * L + l) * D + d;
int halfD = D / 2;  // not `half`: a reserved Metal type name
int base = j - d;
T xv = isq ? q[j] : k[j];
T other = isq ? q[base + (d < halfD ? d + halfD : d - halfD)] : k[base + (d < halfD ? d + halfD : d - halfD)];
T rv = d < halfD ? T(-other) : other;
T a = xv * cos_in[ci];
T s = rv * sin_in[ci];
T y = a + s;
if (isq) { q_out[j] = y; } else { k_out[j] = y; }
