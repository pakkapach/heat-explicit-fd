"""
Explicit finite difference method for the parabolic PDE (1D heat equation)
Kreyszig, Advanced Engineering Mathematics, Sec. 21.6 (numerics) and Sec. 12.6 (exact series)

โจทย์ที่แก้ (สมการความร้อน 1 มิติ):
    u_t = c^2 u_xx,   0 <= x <= L,  t >= 0
    u(x, 0) = f(x)                     เงื่อนไขเริ่มต้น: อุณหภูมิตอนเริ่ม
    u(0, t) = g0(t),  u(L, t) = gL(t)  เงื่อนไขขอบ: อุณหภูมิที่ปลายทั้งสองข้าง (ปกติ = 0)

แบ่งตาราง (grid):
    x_i = i*h   (h = ระยะห่างในแกน x)
    t_j = j*k   (k = ขนาดก้าวเวลา)
    r = c^2 k / h^2

สมการผลต่าง (4): แทนอนุพันธ์ด้วยผลต่าง
    u_t  ~ (u[i,j+1] - u[i,j]) / k                      ผลต่างไปข้างหน้าในเวลา
    u_xx ~ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / h^2         ผลต่างกลางในแกน x

แก้หา u[i,j+1] ได้วิธี explicit (5):
    u[i,j+1] = (1 - 2r) u[i,j] + r (u[i+1,j] + u[i-1,j])
    -> ค่าที่เวลาถัดไป คำนวณจาก 3 จุดในแถวเวลาปัจจุบันได้ตรงๆ ไม่ต้องแก้ระบบสมการ

เงื่อนไขเสถียรภาพ (6):   r <= 1/2
    ถ้า r > 1/2 สัมประสิทธิ์ (1 - 2r) ติดลบ ความคลาดเคลื่อนจะขยายและผลลัพธ์แกว่ง/ผิด

ปลายซ้ายหุ้มฉนวน u_x(0,t) = 0 (ข้อ 10):
    u[0,j+1] = (1 - 2r) u[0,j] + 2r u[1,j]
"""

import numpy as np


# ---------------------------------------------------------------- explicit method (5)
def explicit(f, L=1.0, c=1.0, h=0.2, k=0.01, t_end=0.2,
             g0=lambda t: 0.0, gL=lambda t: 0.0, insulated_left=False):
    """คืนค่า x, t, U, r  โดย U[j, i] = u(x_i, t_j)  (แถว = เวลา, คอลัมน์ = ตำแหน่ง)"""
    n = int(round(L / h))             # จำนวนช่วงในแกน x  -> มีจุด x ทั้งหมด n+1 จุด
    steps = int(round(t_end / k))     # จำนวนก้าวเวลา
    x = np.linspace(0, L, n + 1)      # x_0, x_1, ..., x_n
    t = np.linspace(0, steps * k, steps + 1)
    r = c**2 * k / h**2
    if r > 0.5:
        print(f"WARNING: r = {r:.4g} > 1/2 violates condition (6) -> results are unreliable")

    U = np.zeros((steps + 1, n + 1))
    U[0, :] = f(x)                    # แถวแรก (t = 0) = อุณหภูมิเริ่มต้น f(x)
    if not insulated_left:
        U[0, 0] = g0(0)               # ค่าขอบซ้าย
    U[0, -1] = gL(0)                  # ค่าขอบขวา

    for j in range(steps):            # เดินไปทีละแถวเวลา j -> j+1
        # จุดภายใน i = 1..n-1 ทุกจุดพร้อมกัน ตามสูตร (5):
        #   U[j, 1:-1] = u[i,j],  U[j, 2:] = u[i+1,j],  U[j, :-2] = u[i-1,j]
        U[j + 1, 1:-1] = (1 - 2*r) * U[j, 1:-1] + r * (U[j, 2:] + U[j, :-2])
        if insulated_left:
            # ปลายซ้ายหุ้มฉนวน: ใช้จุดสมมติ u[-1,j] = u[1,j] เพราะ u_x(0,t) = 0
            U[j + 1, 0] = (1 - 2*r) * U[j, 0] + 2*r * U[j, 1]
        else:
            U[j + 1, 0] = g0(t[j + 1])  # กำหนดอุณหภูมิปลายซ้าย
        U[j + 1, -1] = gL(t[j + 1])     # กำหนดอุณหภูมิปลายขวา
    return x, t, U, r


# ---------------------------------------------------------------- exact solution (Sec. 12.6)
def exact_series(f, x, t, L=1.0, c=1.0, terms=50, quad_pts=2001):
    """ค่าจริงจากอนุกรมฟูเรียร์ (ใช้ได้เมื่อปลายทั้งสองข้าง = 0 เท่านั้น)
        u(x,t) = sum B_n sin(n pi x / L) exp(-(c n pi / L)^2 t)
        B_n    = (2/L) int_0^L f(x) sin(n pi x / L) dx      (อินทิเกรตเชิงตัวเลขด้วยกฎสี่เหลี่ยมคางหมู)
    """
    xs = np.linspace(0, L, quad_pts)
    fx = f(xs)
    X, T = np.meshgrid(x, t)
    u = np.zeros_like(X, dtype=float)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    for n in range(1, terms + 1):
        Bn = 2 / L * trapz(fx * np.sin(n*np.pi*xs/L), xs)
        u += Bn * np.sin(n*np.pi*X/L) * np.exp(-(c*n*np.pi/L)**2 * T)
    return u


def print_table(x, t, U, title, every=1):
    """พิมพ์ตาราง u(x,t): แต่ละแถวคือเวลาหนึ่ง  every = พิมพ์ทุกกี่ก้าวเวลา"""
    print(f"\n{title}")
    print("   t    | " + " ".join(f"x={xi:<6.2f}" for xi in x))
    for j in range(0, len(t), every):
        print(f" {t[j]:6.3f} | " + " ".join(f"{v:8.4f}" for v in U[j]))


def show_steps(x, t, U, r):
    """แสดงการแทนค่าในสูตร (5) ทีละจุด เหมือนคิดด้วยมือ"""
    for j in range(len(t) - 1):
        print(f"\n--- j = {j} -> {j+1}   (t = {t[j]:g} -> {t[j+1]:g}) ---")
        for i in range(1, len(x) - 1):
            print(f"u{i},{j+1} = {1-2*r:g}*{U[j,i]:.4f} + {r:g}*({U[j,i+1]:.4f} + {U[j,i-1]:.4f})"
                  f" = {U[j+1,i]:.4f}")


# ---------------------------------------------------------------- examples from Sec. 21.6
if __name__ == "__main__":

    # ---------- Problem 5 (แสดงวิธีทำละเอียด) ----------
    # แท่งยาว L = 10 ft หุ้มฉนวนด้านข้าง, ปลายทั้งสอง = 0, อุณหภูมิเริ่มต้น f(x) = x(1 - 0.1x)
    # ใช้ h = 1, k = 0.5 หาอุณหภูมิที่ t = 2  (c = 1 -> r = 0.5 -> 4 ก้าวเวลา)
    f5 = lambda x: x * (1 - 0.1*x)
    x, t, U, r = explicit(f5, L=10, h=1, k=0.5, t_end=2)
    print(f"Problem 5: r = k/h^2 = {r}  ->  u[i,j+1] = {1-2*r:g}*u[i,j] + {r}*(u[i+1,j] + u[i-1,j])")
    show_steps(x, t, U, r)
    print_table(x, t, U, "Problem 5: explicit (h=1, k=0.5)")
    print_table(x, t, exact_series(f5, x, t, L=10), "Problem 5: exact (Fourier series)")

    # ---------- Example 1: u(x,0) = sin(pi x), h = 0.2, r = 0.25 (k = 0.01), 20 ก้าว ----------
    f = lambda x: np.sin(np.pi * x)
    x, t, U, r = explicit(f, h=0.2, k=0.01, t_end=0.2)
    print_table(x, t, U, f"Example 1: explicit (h=0.2, k=0.01, r={r:.2f}), every 4th step", every=4)
    print_table(x, t, np.sin(np.pi*x) * np.exp(-np.pi**2 * t[:, None]),
                "Example 1: exact u = sin(pi x) exp(-pi^2 t)  (10)", every=4)

    # Example 1 ด้วย r = 1 (ผิดเงื่อนไข (6)) -> ค่าคลาดเคลื่อนมาก
    x, t, U, r = explicit(f, h=0.2, k=0.04, t_end=0.2)
    print_table(x, t, U, f"Example 1 with r={r:.0f} (violates r <= 1/2)")

    # ---------- Problem 6: f(x) = x บน [0, 1/2), 1 - x บน [1/2, 1], h = 0.2, k = 0.01, 8 ก้าว ----------
    # ค่าจากอนุกรมในหนังสือที่ t = 0.08: 0.108 (x=0.2), 0.175 (x=0.4)
    f6 = lambda x: np.where(x < 0.5, x, 1 - x)
    x, t, U, r = explicit(f6, h=0.2, k=0.01, t_end=0.08)
    print_table(x, t, U, f"Problem 6 (h=0.2, k=0.01, r={r:.2f})")
    print_table(x, t, exact_series(f6, x, t), "Problem 6: exact (Fourier series)", every=4)

    # ---------- Problem 7: เหมือนข้อ 6 แต่ r = 1/2 (k = 0.02), 4 ก้าว ----------
    # ค่าในหนังสือ: 0.156, 0.254 (t=0.04); 0.105, 0.170 (t=0.08)
    x, t, U, r = explicit(f6, h=0.2, k=0.02, t_end=0.08)
    print_table(x, t, U, f"Problem 7 (h=0.2, k=0.02, r={r:.2f})")

    # ---------- Problem 9: f(x) = x บน [0, 0.2], 0.25(1 - x) บน (0.2, 1], h = 0.2, k = 0.01, 5 ก้าว ----------
    f9 = lambda x: np.where(x <= 0.2, x, 0.25*(1 - x))
    x, t, U, r = explicit(f9, h=0.2, k=0.01, t_end=0.05)
    print_table(x, t, U, f"Problem 9 (h=0.2, k=0.01, r={r:.2f})")

    # ---------- Problem 10: ปลายซ้ายหุ้มฉนวน, ปลายขวา u(1,t) = sin(50 pi t / 3), u(x,0) = 0 ----------
    x, t, U, r = explicit(lambda x: 0*x, h=0.2, k=0.01, t_end=0.12,
                          gL=lambda t: np.sin(50*np.pi*t/3), insulated_left=True)
    print_table(x, t, U, f"Problem 10: insulated left end (h=0.2, k=0.01, r={r:.2f})")

    # ---------- กราฟ (ถ้ามี matplotlib) ----------
    try:
        import matplotlib.pyplot as plt
        x, t, U, r = explicit(f5, L=10, h=1, k=0.5, t_end=2)
        for j in range(len(t)):
            plt.plot(x, U[j], "o-", label=f"t={t[j]:g}")
        plt.xlabel("x (ft)"); plt.ylabel("u(x,t)")
        plt.title("Problem 5: explicit method, h=1, k=0.5")
        plt.legend(); plt.grid(True); plt.show()
    except ImportError:
        pass
