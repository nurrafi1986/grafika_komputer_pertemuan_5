import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PUSTAKA FUNGSI TRANSFORMASI (3x3)
# ==========================================

def get_translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)

def get_scaling_matrix(sx, sy):
    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=float)

def get_rotation_matrix(angle_degrees):
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)

def get_shear_matrix(shx=0.0, shy=0.0):
    return np.array([
        [1,   shx, 0],
        [shy, 1,   0],
        [0,   0,   1]
    ], dtype=float)

def get_reflection_matrix(axis='Y'):
    if axis.upper() == 'Y':
        return np.array([
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], dtype=float)
    elif axis.upper() == 'X':
        return np.array([
            [1,  0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=float)
    elif axis.lower() == 'origin':
        return np.array([
            [-1,  0, 0],
            [ 0, -1, 0],
            [ 0,  0, 1]
        ], dtype=float)
    elif axis.lower() == 'y=x':
        return np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)
    else:
        raise ValueError("Sumbu refleksi tidak valid. Pilih 'X', 'Y', 'origin', atau 'y=x'.")

# ==========================================
# 2. FUNGSI APLIKASI & KOMPOSISI TRANSFORMASI
# ==========================================

def apply_transformation(matrix, vertices):
    """
    Mengalikan matriks transformasi 3x3 dengan verteks Nx2 (Homogen 3xN)
    """
    num_pts = vertices.shape[0]
    homo_vertices = np.hstack((vertices, np.ones((num_pts, 1)))).T
    transformed_homo = np.dot(matrix, homo_vertices)
    return transformed_homo[:2, :].T

def transformasi_terhadap_pivot(vertices, sudut_derajat, xp, yp):
    """
    Membuat matriks komposit T(xp, yp) * R(theta) * T(-xp, -yp)
    """
    T_to_origin = get_translation_matrix(-xp, -yp)
    R = get_rotation_matrix(sudut_derajat)
    T_back = get_translation_matrix(xp, yp)
    
    # Komposisi Matriks (dari kanan ke kiri)
    M_pivot = np.dot(T_back, np.dot(R, T_to_origin))
    return apply_transformation(M_pivot, vertices)

# ==========================================
# 3. STUDI KASUS VISUALIZATION (MATPLOTLIB)
# ==========================================

if __name__ == "__main__":
    # Definisikan bangun geometri Panah Navigasi (5 verteks)
    objek_asli = np.array([
        [2.0, 1.0],
        [4.0, 1.0],
        [4.0, 3.0],
        [5.0, 3.0],
        [3.0, 5.0],
        [1.0, 3.0],
        [2.0, 3.0]
    ])

    # Hitung Centroid / Titik Berat (x_bar, y_bar)
    centroid_x = np.mean(objek_asli[:, 0])
    centroid_y = np.mean(objek_asli[:, 1])

    # --- Subplot 1: Rotasi Poros Sendiri (Centroid) ---
    objek_rotasi = transformasi_terhadap_pivot(objek_asli, 60, centroid_x, centroid_y)

    # --- Subplot 2: Shear Horisontal (shx=0.75) dilanjutkan Refleksi Sumbu Y ---
    M_shear = get_shear_matrix(shx=0.75, shy=0)
    M_ref = get_reflection_matrix('Y')
    M_komposit_s2 = np.dot(M_ref, M_shear)  # Shear dulu baru Refleksi
    objek_shear_ref = apply_transformation(M_komposit_s2, objek_asli)

    # Fungsi pembantu untuk menutup kurva poligon saat di-plot
    def close_polygon(pts):
        return np.vstack([pts, pts[0]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot Subplot 1 ---
    poly_asli = close_polygon(objek_asli)
    poly_rotasi = close_polygon(objek_rotasi)

    ax1.plot(poly_asli[:, 0], poly_asli[:, 1], 'b-o', label='Objek Asli', linewidth=1.5)
    ax1.plot(poly_rotasi[:, 0], poly_rotasi[:, 1], 'r--s', label='Rotasi 60° (Centroid)', linewidth=1.5)
    ax1.plot(centroid_x, centroid_y, 'ro', markersize=8, label=f'Centroid ({centroid_x:.2f}, {centroid_y:.2f})')
    
    ax1.set_title('Subplot 1: Rotasi Terhadap Titik Centroid')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis('equal')
    ax1.legend(loc='best')

    # --- Plot Subplot 2 ---
    poly_shear_ref = close_polygon(objek_shear_ref)

    ax2.plot(poly_asli[:, 0], poly_asli[:, 1], 'b-o', label='Objek Asli', linewidth=1.5)
    ax2.plot(poly_shear_ref[:, 0], poly_shear_ref[:, 1], 'g--^', label='Shear (shx=0.75) + Refleksi Y', linewidth=1.5)

    ax2.set_title('Subplot 2: Shear Horisontal dilanjutkan Refleksi Sumbu Y')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axis('equal')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()