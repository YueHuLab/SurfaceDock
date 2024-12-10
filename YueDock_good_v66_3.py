import numpy as np
from scipy.spatial import KDTree
from Bio.PDB import PDBParser, PDBIO, Superimposer
import plyfile
import os
from scipy.spatial.transform import Rotation as R


# Load the mesh and extract physical-chemical features from the PLY file
def load_ply_features(ply_file_path):
    """
    从 .ply 文件中提取物理化学特征 (charge, hbond, hphob, iface)
    参数：
        ply_file_path: .ply 文件路径
    返回：
        mesh: 顶点的坐标数据
        features: 包含 (charge, hbond, hphob, iface) 的特征字典
    """
    plydata = plyfile.PlyData.read(ply_file_path)
    vertex = plydata['vertex']
    mesh = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    features = {
        'charge': np.array(vertex['charge']),
        'hbond': np.array(vertex['hbond']),
        'hphob': np.array(vertex['hphob']),
        'iface': np.array(vertex['iface'])
    }
    return mesh, features

# Function to extract seed points using second-order partial derivative test
def find_seed_points_f11(vertices, con, KD_threshold=45, threshold=1):
    # Use KDTree for finding neighbors
    kdtree = KDTree(vertices)
    seed_points = []
    print(f"KDTree vertices size  {len(vertices)}")
    for i, point in enumerate(vertices):
        # Find neighboring points within a distance threshold
        neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)
        #print(f"neighbors size  {len(neighbors_idx)}")
        neighbors = vertices[neighbors_idx]
		
        if len(neighbors) < 3:
            continue
            
        # Calculate second-order partial derivatives using FDM
        f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = calculate_second_order_derivatives(point, neighbors)
        D1 = f_xx
        D2 = np.linalg.det([[f_xx, f_xy], [f_yx, f_yy]])
        D3 = np.linalg.det([[f_xx, f_xy, f_xz], [f_yx, f_yy, f_yz], [f_zx, f_zy, f_zz]])
            
        # Identify local maxima or minima based on second-order partial derivative test
        if con==0 and D1 > 0 and D2 > 0 and D3 > 0:
            seed_points.append(point)  # Local minimum
        elif con==1 and D1 < 0 and D2 > 0 and D3 < 0:
            seed_points.append(point)  # Local maximum
    return np.array(seed_points)

def find_seed_points_f12(vertices, con, KD_threshold=20, threshold=1):
    # Use KDTree for finding neighbors
    kdtree = KDTree(vertices)
    seed_points = []
    print(f"KDTree vertices size: {len(vertices)}")
    
    # 计算质心
    centroid = np.mean(vertices, axis=0)

    for i, point in enumerate(vertices):
        # Find neighboring points within a distance threshold
        neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)
        neighbors = vertices[neighbors_idx]
        
        if len(neighbors) < 3:
            continue
        
        # 计算二阶偏导数
        f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = calculate_second_order_derivatives(point, neighbors)
        
        # 构造海森矩阵
        H = np.array([[f_xx, f_xy, f_xz],
                      [f_yx, f_yy, f_yz],
                      [f_zx, f_zy, f_zz]])


        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(H)
        D1 = f_xx
        D2 = np.linalg.det([[f_xx, f_xy], [f_yx, f_yy]])
        D3 = np.linalg.det(H)
        # 排序特征值和对应的特征向量
        sorted_indices = np.argsort(eigenvalues)  # 获取排序索引
        sorted_eigenvalues = eigenvalues[sorted_indices]  # 排序特征值
        sorted_eigenvectors = eigenvectors[:, sorted_indices]  # 根据索引排序特征向量

        # 输出排序后的特征值和特征向量
        print(f"排序后的特征值: {sorted_eigenvalues}")
        print("对应的特征向量:")
        print(sorted_eigenvectors)
        Switch=0
        if np.all(sorted_eigenvalues > 0) or np.all(sorted_eigenvalues < 0):
            Switch=1

        # 计算点与质心的距离
        distance_to_centroid = np.linalg.norm(point - centroid)

        # 计算邻居与质心的平均距离
        if len(neighbors) > 0:
            avg_distance_neighbors = np.mean([np.linalg.norm(neighbor - centroid) for neighbor in neighbors])
        else:
            avg_distance_neighbors = 0
        
        # 判断凹凸性
        #if distance_to_centroid > avg_distance_neighbors:
        #   print(f"Point {point} is convex.")
        #else:
        #   print(f"Point {point} is concave.")
        # Identify local maxima or minima based on second-order partial derivative test
		
        if con==0 and Switch==1 and distance_to_centroid < avg_distance_neighbors:
            seed_points.append(point)  # Local minimum
            print("mini ")            
        elif con==1 and  Switch==1 and distance_to_centroid > avg_distance_neighbors:
            seed_points.append(point)  # Local maximum
            print("ok max")
        #if i==10:
        #    break			

    return np.array(seed_points)

def find_seed_points_f15(vertices, con, KD_threshold=20, threshold=1):
    # Use KDTree for finding neighbors
    kdtree = KDTree(vertices)
    seed_points = []
    avg_distances = []  # To store average distances of neighbors to the centroid
    abs_differences = []  # To store absolute differences
    print(f"KDTree vertices size: {len(vertices)}")
    
    # Calculate the centroid
    centroid = np.mean(vertices, axis=0)

    for i, point in enumerate(vertices):
        # Find neighboring points within a distance threshold
        neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)
        neighbors = vertices[neighbors_idx]
        
        if len(neighbors) < 3:
            continue
        
        # Calculate second-order derivatives
        f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = calculate_second_order_derivatives(point, neighbors)
        
        # Construct Hessian matrix
        H = np.array([[f_xx, f_xy, f_xz],
                      [f_yx, f_yy, f_yz],
                      [f_zx, f_zy, f_zz]])

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(H)

        Switch = 0
        if np.all(eigenvalues > 0) or np.all(eigenvalues < 0):
            Switch = 1

        # Calculate distance to centroid
        distance_to_centroid = np.linalg.norm(point - centroid)

        # Calculate average distance to centroid for neighbors
        avg_distance_neighbors = np.mean([np.linalg.norm(neighbor - centroid) for neighbor in neighbors]) if len(neighbors) > 0 else 0
        
        # Calculate absolute difference
        abs_difference = abs(distance_to_centroid - avg_distance_neighbors)

        # Identify local maxima or minima based on second-order partial derivative test
        if con == 0 and Switch == 1 and distance_to_centroid < avg_distance_neighbors:
            seed_points.append(point)  # Local minimum
            avg_distances.append(avg_distance_neighbors)
            abs_differences.append(abs_difference)
            print("mini")
        elif con == 1 and Switch == 1 and distance_to_centroid > avg_distance_neighbors:
            seed_points.append(point)  # Local maximum
            avg_distances.append(avg_distance_neighbors)
            abs_differences.append(abs_difference)
            print("ok max")

    # Convert seed_points to a NumPy array for easier manipulation
    seed_points = np.array(seed_points)
    avg_distances = np.array(avg_distances)
    abs_differences = np.array(abs_differences)

    # Sort seed points based on absolute differences in descending order
    sorted_indices = np.argsort(-abs_differences)  # Negative sign for descending order
    sorted_seed_points = seed_points[sorted_indices]

    return sorted_seed_points




def find_seed_points(vertices, con, KD_threshold=20, threshold=1):
    kdtree = KDTree(vertices)
    seed_points = []
    avg_distances = []  # To store average distances of neighbors to the centroid
    abs_differences = []  # To store absolute differences
    eigenvalues_list = []  # To store eigenvalues
    eigenvectors_list = []  # To store eigenvectors
    
    print(f"KDTree vertices size: {len(vertices)}")
    centroid = np.mean(vertices, axis=0)

    for i, point in enumerate(vertices):
        neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)
        neighbors = vertices[neighbors_idx]
        
        if len(neighbors) < 3:
            continue
        
        f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = calculate_second_order_derivatives(point, neighbors)
        
        H = np.array([[f_xx, f_xy, f_xz],
                      [f_yx, f_yy, f_yz],
                      [f_zx, f_zy, f_zz]])

        eigenvalues, eigenvectors = np.linalg.eig(H)
        
        Switch = 0
        if np.all(eigenvalues > 0) or np.all(eigenvalues < 0):
            Switch = 1

        distance_to_centroid = np.linalg.norm(point - centroid)
        avg_distance_neighbors = np.mean([np.linalg.norm(neighbor - centroid) for neighbor in neighbors]) if len(neighbors) > 0 else 0
        abs_difference = abs(distance_to_centroid - avg_distance_neighbors)

        if (con == 0 and Switch == 1 and distance_to_centroid < avg_distance_neighbors) or \
           (con == 1 and Switch == 1 and distance_to_centroid > avg_distance_neighbors):
            seed_points.append(point)
            avg_distances.append(avg_distance_neighbors)
            abs_differences.append(abs_difference)
            eigenvalues_list.append(eigenvalues)
            eigenvectors_list.append(eigenvectors)
            print("mini" if con == 0 else "ok max")

    seed_points = np.array(seed_points)
    avg_distances = np.array(avg_distances)
    abs_differences = np.array(abs_differences)
    eigenvalues_list = np.array(eigenvalues_list)
    eigenvectors_list = np.array(eigenvectors_list)

    sorted_indices = np.argsort(-abs_differences)
    sorted_seed_points = seed_points[sorted_indices]
    sorted_abs_differences = abs_differences[sorted_indices]
    sorted_eigenvalues = eigenvalues_list[sorted_indices]
    sorted_eigenvectors = eigenvectors_list[sorted_indices]

    return sorted_seed_points, sorted_abs_differences, sorted_eigenvalues, sorted_eigenvectors
	
	
def find_seed_points_energy(vertices, receptor_features,con, KD_threshold=15, threshold=1):
    kdtree = KDTree(vertices)
    seed_points = []
    avg_distances = []  # To store average distances of neighbors to the centroid
    abs_differences = []  # To store absolute differences
    eigenvalues_list = []  # To store eigenvalues
    eigenvectors_list = []  # To store eigenvectors
    
    print(f"KDTree vertices size: {len(vertices)}")
    centroid = np.mean(vertices, axis=0)

    for i, point in enumerate(vertices):
        neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)
		#debug 1105
        neighbors_idx = [idx for idx in neighbors_idx if not np.array_equal(vertices[idx], point)]		
        neighbors = vertices[neighbors_idx]
        neighbors_charges = receptor_features['charge'][neighbors_idx]
        neighbors_hbonds = receptor_features['hbond'][neighbors_idx]
        neighbors_hphob = receptor_features['hphob'][neighbors_idx]  

        point_idx = kdtree.query(point)[1]
        point_idx_charges = receptor_features['charge'][point_idx]
        point_idx_hbonds = receptor_features['hbond'][point_idx]
        point_idx_hphob = receptor_features['hphob'][point_idx]  		
		
        if len(neighbors) < 3:
            continue
        #for h in (0.001,0.01,0.05,0.1,0.5,1):
        f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy = calculate_second_order_derivatives_energy(point, neighbors,  
                                       neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                       point_idx_charges, point_idx_hbonds, point_idx_hphob
                                       )
            #print("h",h)
        H = np.array([[f_xx, f_xy, f_xz],
                      [f_yx, f_yy, f_yz],
                      [f_zx, f_zy, f_zz]])

        eigenvalues, eigenvectors = np.linalg.eig(H)
        print("eigenvalues ",eigenvalues)
        Switch = 0
        if np.all(eigenvalues > 0) or np.all(eigenvalues < 0):
            Switch = 1

        distance_to_centroid = np.linalg.norm(point - centroid)
        avg_distance_neighbors = np.mean([np.linalg.norm(neighbor - centroid) for neighbor in neighbors]) if len(neighbors) > 0 else 0
        abs_difference = abs(distance_to_centroid - avg_distance_neighbors)

        if (con == 0 and Switch == 1 and distance_to_centroid < avg_distance_neighbors) or \
           (con == 1 and Switch == 1 and distance_to_centroid > avg_distance_neighbors):
            seed_points.append(point)
            avg_distances.append(avg_distance_neighbors)
            abs_differences.append(abs_difference)
            eigenvalues_list.append(eigenvalues)
            eigenvectors_list.append(eigenvectors)
            print("mini" if con == 0 else "ok max")
        #if i==100:
        #   break
    seed_points = np.array(seed_points)
    avg_distances = np.array(avg_distances)
    abs_differences = np.array(abs_differences)
    eigenvalues_list = np.array(eigenvalues_list)
    eigenvectors_list = np.array(eigenvectors_list)

    sorted_indices = np.argsort(-abs_differences)
    sorted_seed_points = seed_points[sorted_indices]
    sorted_abs_differences = abs_differences[sorted_indices]
    sorted_eigenvalues = eigenvalues_list[sorted_indices]
    sorted_eigenvectors = eigenvectors_list[sorted_indices]

    return sorted_seed_points, sorted_abs_differences, sorted_eigenvalues, sorted_eigenvectors	
	
def calculate_second_order_derivatives_energy(point, neighbors,
                                       neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                       point_idx_charges, point_idx_hbonds, point_idx_hphob, 
                                       h=0.05):
    """
    This function calculates the second-order partial derivatives of the energy using finite differences.
    
    Parameters:
    - point: The coordinates of the point (numpy array)
    - neighbors: The coordinates of the neighboring points (numpy array)
    - receptor_features: A dictionary containing the receptor features ('charge', 'hbond', 'hphob')
    - kdtree: A KDTree for neighbor search
    - neighbors_charges: The charges of the neighbors (numpy array)
    - neighbors_hbonds: The hydrogen bond features of the neighbors (numpy array)
    - neighbors_hphob: The hydrophobicity features of the neighbors (numpy array)
    - point_idx_charges: The charge of the point (float)
    - point_idx_hbonds: The hydrogen bond feature of the point (float)
    - point_idx_hphob: The hydrophobicity feature of the point (float)
    - h: The step size for finite difference calculations
    
    Returns:
    - Second-order partial derivatives of the energy with respect to the x, y, and z coordinates.
    """
    
    # Finite difference calculation for second-order derivatives
    f_xx = (evaluate_function(point + np.array([h, 0, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            2 * evaluate_function(point, neighbors, 
                                  neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                  point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([h, 0, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (h ** 2)

    f_yy = (evaluate_function(point + np.array([0, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            2 * evaluate_function(point, neighbors, 
                                  neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                  point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([0, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (h ** 2)

    f_zz = (evaluate_function(point + np.array([0, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            2 * evaluate_function(point, neighbors, 
                                  neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                  point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([0, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (h ** 2)

    f_xy = (evaluate_function(point + np.array([h, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([h, -h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point - np.array([h, -h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([h, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_yx = f_xy  # Assuming mixed partial derivatives are equal (Schwarz's theorem)

    f_xz = (evaluate_function(point + np.array([h, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([h, 0, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point - np.array([h, 0, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([h, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_zx = f_xz  # Assuming mixed partial derivatives are equal (Schwarz's theorem)

    f_yz = (evaluate_function(point + np.array([0, h, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([0, h, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point - np.array([0, h, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([0, h, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_zy = f_yz  # Assuming mixed partial derivatives are equal (Schwarz's theorem)
    print("f_xx ",f_xx)
    return f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy
# 记得定义 calculate_second_order_derivatives 函数	
import numpy as np

def calculate_second_order_derivatives_energy_test(point, neighbors,
                                       neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                       point_idx_charges, point_idx_hbonds, point_idx_hphob, 
                                       h=0.05):
    """
    This function calculates the second-order partial derivatives of the energy using finite differences.
    
    Parameters:
    - point: The coordinates of the point (numpy array)
    - neighbors: The coordinates of the neighboring points (numpy array)
    - neighbors_charges: The charges of the neighbors (numpy array)
    - neighbors_hbonds: The hydrogen bond features of the neighbors (numpy array)
    - neighbors_hphob: The hydrophobicity features of the neighbors (numpy array)
    - point_idx_charges: The charge of the point (float)
    - point_idx_hbonds: The hydrogen bond feature of the point (float)
    - point_idx_hphob: The hydrophobicity feature of the point (float)
    - h: The step size for finite difference calculations
    
    Returns:
    - All second-order partial derivatives of the energy with respect to the x, y, and z coordinates.
    """
    
    # Second-order partial derivatives along the main axes
    f_xx = (evaluate_function(point + np.array([h, 0, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            2 * evaluate_function(point, neighbors, 
                                  neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                  point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([h, 0, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (h ** 2)

    f_yy = (evaluate_function(point + np.array([0, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            2 * evaluate_function(point, neighbors, 
                                  neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                  point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([0, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (h ** 2)

    f_zz = (evaluate_function(point + np.array([0, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            2 * evaluate_function(point, neighbors, 
                                  neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                                  point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([0, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (h ** 2)

    # Mixed second-order partial derivatives
    f_xy = (evaluate_function(point + np.array([h, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([h, -h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point - np.array([h, -h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([h, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_yx = (evaluate_function(point + np.array([h, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([-h, h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([-h, -h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point + np.array([h, -h, 0]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_xz = (evaluate_function(point + np.array([h, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([h, 0, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point - np.array([h, 0, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([h, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_zx = (evaluate_function(point + np.array([h, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([-h, 0, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([-h, 0, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point + np.array([h, 0, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_yz = (evaluate_function(point + np.array([0, h, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([0, h, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point - np.array([0, h, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point - np.array([0, h, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)

    f_zy = (evaluate_function(point + np.array([0, h, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([0, -h, h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) -
            evaluate_function(point + np.array([0, -h, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob) +
            evaluate_function(point + np.array([0, h, -h]), neighbors, 
                              neighbors_charges, neighbors_hbonds, neighbors_hphob, 
                              point_idx_charges, point_idx_hbonds, point_idx_hphob)) / (4 * h ** 2)
    f_xy=(f_xy+f_yx)/2.0
    f_yx=f_xy
    f_xz=(f_xz+f_zx)/2.0
    f_zx=f_xz
    f_zy=(f_zy+f_yz)/2.0
    f_yz=f_zy
			
	
	
    return f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy



def calculate_second_order_derivatives(point, neighbors, h=0.01):
    # Approximate second-order partial derivatives using Finite Differences Method (FDM)
    f_xx = (evaluate_function(point + np.array([h, 0, 0]), neighbors) -
            2 * evaluate_function(point, neighbors) +
            evaluate_function(point - np.array([h, 0, 0]), neighbors)) / (h ** 2)

    f_yy = (evaluate_function(point + np.array([0, h, 0]), neighbors) -
            2 * evaluate_function(point, neighbors) +
            evaluate_function(point - np.array([0, h, 0]), neighbors)) / (h ** 2)

    f_zz = (evaluate_function(point + np.array([0, 0, h]), neighbors) -
            2 * evaluate_function(point, neighbors) +
            evaluate_function(point - np.array([0, 0, h]), neighbors)) / (h ** 2)

    f_xy = (evaluate_function(point + np.array([h, h, 0]), neighbors) -
            evaluate_function(point + np.array([h, -h, 0]), neighbors) -
            evaluate_function(point - np.array([h, -h, 0]), neighbors) +
            evaluate_function(point - np.array([h, h, 0]), neighbors)) / (4 * h ** 2)

    f_yx = f_xy  # Assuming mixed partial derivatives are equal (Schwarz's theorem)

    f_xz = (evaluate_function(point + np.array([h, 0, h]), neighbors) -
            evaluate_function(point + np.array([h, 0, -h]), neighbors) -
            evaluate_function(point - np.array([h, 0, -h]), neighbors) +
            evaluate_function(point - np.array([h, 0, h]), neighbors)) / (4 * h ** 2)

    f_zx = f_xz  # Assuming mixed partial derivatives are equal (Schwarz's theorem)

    f_yz = (evaluate_function(point + np.array([0, h, h]), neighbors) -
            evaluate_function(point + np.array([0, h, -h]), neighbors) -
            evaluate_function(point - np.array([0, h, -h]), neighbors) +
            evaluate_function(point - np.array([0, h, h]), neighbors)) / (4 * h ** 2)

    f_zy = f_yz  # Assuming mixed partial derivatives are equal (Schwarz's theorem)

    return f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy


import numpy as np

def calculate_second_order_derivatives_f9(point, neighbors, h=0.01):
    # Approximate second-order partial derivatives using Finite Differences Method (FDM)
    
    # Second-order partial derivatives
    f_xx = (evaluate_function(point + np.array([h, 0, 0]), neighbors) -
            2 * evaluate_function(point, neighbors) +
            evaluate_function(point - np.array([h, 0, 0]), neighbors)) / (h ** 2)

    f_yy = (evaluate_function(point + np.array([0, h, 0]), neighbors) -
            2 * evaluate_function(point, neighbors) +
            evaluate_function(point - np.array([0, h, 0]), neighbors)) / (h ** 2)

    f_zz = (evaluate_function(point + np.array([0, 0, h]), neighbors) -
            2 * evaluate_function(point, neighbors) +
            evaluate_function(point - np.array([0, 0, h]), neighbors)) / (h ** 2)

    # Mixed partial derivatives (no symmetry assumptions)
    f_xy = (evaluate_function(point + np.array([h, h, 0]), neighbors) -
            evaluate_function(point + np.array([h, -h, 0]), neighbors) -
            evaluate_function(point - np.array([h, -h, 0]), neighbors) +
            evaluate_function(point - np.array([h, h, 0]), neighbors)) / (4 * h ** 2)

    f_yx = (evaluate_function(point + np.array([h, h, 0]), neighbors) -
            evaluate_function(point + np.array([-h, h, 0]), neighbors) -
            evaluate_function(point + np.array([-h, -h, 0]), neighbors) +
            evaluate_function(point + np.array([h, -h, 0]), neighbors)) / (4 * h ** 2)

    f_xz = (evaluate_function(point + np.array([h, 0, h]), neighbors) -
            evaluate_function(point + np.array([h, 0, -h]), neighbors) -
            evaluate_function(point - np.array([h, 0, -h]), neighbors) +
            evaluate_function(point - np.array([h, 0, h]), neighbors)) / (4 * h ** 2)

    f_zx = (evaluate_function(point + np.array([h, 0, h]), neighbors) -
            evaluate_function(point + np.array([-h, 0, h]), neighbors) -
            evaluate_function(point + np.array([-h, 0, -h]), neighbors) +
            evaluate_function(point + np.array([h, 0, -h]), neighbors)) / (4 * h ** 2)

    f_yz = (evaluate_function(point + np.array([0, h, h]), neighbors) -
            evaluate_function(point + np.array([0, h, -h]), neighbors) -
            evaluate_function(point - np.array([0, h, -h]), neighbors) +
            evaluate_function(point - np.array([0, h, h]), neighbors)) / (4 * h ** 2)

    f_zy = (evaluate_function(point + np.array([0, h, h]), neighbors) -
            evaluate_function(point + np.array([0, -h, h]), neighbors) -
            evaluate_function(point + np.array([0, -h, -h]), neighbors) +
            evaluate_function(point + np.array([0, h, -h]), neighbors)) / (4 * h ** 2)

    return f_xx, f_yy, f_zz, f_xy, f_yx, f_xz, f_zx, f_yz, f_zy

def evaluate_function_f1(point, neighbors):
    # Placeholder for evaluating the function value at a point
    # Here we simply return the distance to the centroid of neighbors as a proxy for surface height
    centroid = np.mean(neighbors, axis=0)
    return np.linalg.norm(point - centroid)

import numpy as np

def evaluate_function(point, neighbors_positions,neighbors_charges, neighbors_hbonds, neighbors_hphob,point_charge, point_hbond, point_hphob):
    """
    Calculate the combined interaction score between a specific point and its neighbors.
    
    Parameters:
        point: ndarray
            The 3D coordinates of the point.
        point_charge: float
            Charge of the point.
        point_hbond: float
            Hydrogen bond property of the point.
        point_hphob: float
            Hydrophobic property of the point.
        neighbors_positions: ndarray
            Array of 3D coordinates for neighboring points.
        neighbors_charges: ndarray
            Array of charge values for neighboring points.
        neighbors_hbonds: ndarray
            Array of hydrogen bond values for neighboring points.
        neighbors_hphob: ndarray
            Array of hydrophobic values for neighboring points.
    
    Returns:
        float: The total combined interaction score.
    """

    # Prepare single point inputs as lists for compatibility with the scoring functions
    receptor_positions = [point]
    receptor_charges = [point_charge]
    receptor_hbonds = [point_hbond]
    receptor_hphob = [point_hphob]

    # Call each scoring function with the appropriate parameters
    #charge_score = calculate_charge_score(receptor_positions, receptor_charges, neighbors_positions, neighbors_charges)
    #hbond_score = calculate_hbond_score(receptor_positions, receptor_hbonds, neighbors_positions, neighbors_hbonds)
    #hydrophobicity_score = calculate_hydrophobicity_score(receptor_positions, receptor_hphob, neighbors_positions, neighbors_hphob)
    clash_score = calculate_clash_score_energy(receptor_positions, neighbors_positions)

    # Sum up the scores to get the combined score
    #total_score = charge_score + hbond_score + hydrophobicity_score
    total_score = clash_score	
    #print("total score" ,total_score)
    return total_score
	
	
	
# Function to calculate surface descriptor for a given point
# Function to calculate surface descriptor for a given point
def calculate_surface_descriptor(point, vertices, KD_threshold=9):
    """
    Calculate the surface descriptor for a given point using its neighbors.

    Parameters:
        point (ndarray): Coordinates of the point (x, y, z).
        vertices (ndarray): Array of all vertices coordinates.
        KD_threshold (float): The radius within which neighbors are considered for descriptor calculations.

    Returns:
        tuple: Surface variation and the eigenvector corresponding to the smallest eigenvalue.
    """
    # Use KDTree to find neighbors within the given KD_threshold
    kdtree = KDTree(vertices)
    neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)
    neighbors = vertices[neighbors_idx]

    # Calculate the covariance matrix for the neighbors
    covariance_matrix = np.cov(neighbors - point, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    min_eigenvalue_index = np.argmin(eigenvalues)
    surface_variation = eigenvalues[min_eigenvalue_index] / eigenvalues.sum()
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    return surface_variation, min_eigenvector

# New function to find the points with max and min charge, hphob, hbond within a threshold

def find_extreme_properties(point, ply_filename, KD_threshold=9):
    """
    Find the points with maximum and minimum values of charge, hphob, and hbond within a threshold distance from the given point.

    Parameters:
        point (ndarray): Coordinates of the point (x, y, z).
        ply_filename (str): Path to the PLY file containing mesh data.
        KD_threshold (float): The radius within which neighbors are considered.

    Returns:
        dict: A dictionary containing the maximum and minimum values for charge, hphob, hbond and their respective points.
    """
    # Load the PLY file using the plyfile library
    ply_data = plyfile.PlyData.read(ply_filename)

    # Check if vertex element is available
    if 'vertex' not in ply_data:
        raise ValueError("PLY file must contain vertex data.")

    vertices = ply_data['vertex']
    coordinates = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # Use KDTree to find neighbors within the given KD_threshold
    kdtree = KDTree(coordinates)
    neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)

    if len(neighbors_idx) == 0:
        return None  # No neighbors found within the threshold

    # Extract relevant properties (charge, hphob, hbond) for the neighbors
    charges = vertices['charge'][neighbors_idx]
    hphobs = vertices['hphob'][neighbors_idx]
    hbonds = vertices['hbond'][neighbors_idx]

    # Find the maximum and minimum values along with the corresponding points
    result = {
        'charge': {
            'max_value': np.max(charges),
            'max_point': coordinates[neighbors_idx[np.argmax(charges)]],
            'min_value': np.min(charges),
            'min_point': coordinates[neighbors_idx[np.argmin(charges)]],
        },
        'hphob': {
            'max_value': np.max(hphobs),
            'max_point': coordinates[neighbors_idx[np.argmax(hphobs)]],
            'min_value': np.min(hphobs),
            'min_point': coordinates[neighbors_idx[np.argmin(hphobs)]],
        },
        'hbond': {
            'max_value': np.max(hbonds),
            'max_point': coordinates[neighbors_idx[np.argmax(hbonds)]],
            'min_value': np.min(hbonds),
            'min_point': coordinates[neighbors_idx[np.argmin(hbonds)]],
        }
    }

    return result	

# BATCH	
# Batch version of calculating surface descriptors for multiple seed points
def calculate_surface_descriptors_batch(points, vertices, KD_threshold=9):
    """
    Calculate the surface descriptors for multiple seed points.

    Parameters:
        seed_points_indices (list[int]): List of indices for the seed points.
        vertices (ndarray): Array of all vertices coordinates.
        KD_threshold (float): The radius within which neighbors are considered for descriptor calculations.

    Returns:
        list[dict]: A list of dictionaries containing surface variation and eigenvector information for each seed point.
    """
    descriptors = []
    for point in points:
        # Using the point directly
        point = point
        surface_variation, min_eigenvector = calculate_surface_descriptor(point, vertices, KD_threshold)
        descriptors.append({
            'point': point,
            'surface_variation': surface_variation,
            'min_eigenvector': min_eigenvector
        })
    return descriptors

# Batch version of finding extreme properties for multiple points
def find_extreme_properties_batch(points, ply_filename, KD_threshold=9):
    """
    Find the points with maximum and minimum values of charge, hphob, and hbond within a threshold distance for multiple seed points.

    Parameters:
        seed_points_indices (list[int]): List of indices for the seed points.
        ply_filename (str): Path to the PLY file containing mesh data.
        KD_threshold (float): The radius within which neighbors are considered.

    Returns:
        list[dict]: A list of dictionaries containing the extreme properties for each seed point.
    """
    # Load the PLY file using the plyfile library
    ply_data = plyfile.PlyData.read(ply_filename)

    # Check if vertex element is available
    if 'vertex' not in ply_data:
        raise ValueError("PLY file must contain vertex data.")

    vertices = ply_data['vertex']
    coordinates = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    results = []
    for point in points:
        # Using the point directly
        point = point
        # Use KDTree to find neighbors within the given KD_threshold
        kdtree = KDTree(coordinates)
        neighbors_idx = kdtree.query_ball_point(point, r=KD_threshold)

        if len(neighbors_idx) == 0:
            continue  # No neighbors found within the threshold

        # Extract relevant properties (charge, hphob, hbond) for the neighbors
        charges = vertices['charge'][neighbors_idx]
        hphobs = vertices['hphob'][neighbors_idx]
        hbonds = vertices['hbond'][neighbors_idx]

        # Find the maximum and minimum values along with the corresponding points
        result = {
            'index': neighbors_idx,
            'charge': {
                'max_value': np.max(charges),
                'max_point': coordinates[neighbors_idx[np.argmax(charges)]],
                'min_value': np.min(charges),
                'min_point': coordinates[neighbors_idx[np.argmin(charges)]],
            },
            'hphob': {
                'max_value': np.max(hphobs),
                'max_point': coordinates[neighbors_idx[np.argmax(hphobs)]],
                'min_value': np.min(hphobs),
                'min_point': coordinates[neighbors_idx[np.argmin(hphobs)]],
            },
            'hbond': {
                'max_value': np.max(hbonds),
                'max_point': coordinates[neighbors_idx[np.argmax(hbonds)]],
                'min_value': np.min(hbonds),
                'min_point': coordinates[neighbors_idx[np.argmin(hbonds)]],
            }
        }
        results.append(result)

    return results

def obtain_binding_poses_f3(receptor_vertices, ligand_vertices, receptor_point, ligand_seed_point, receptor_descriptor, ligand_descriptor, receptor_extreme, ligand_extreme):
    """
    Calculate the binding poses for receptor and ligand.

    Parameters:
        receptor_vertices (ndarray): Coordinates of the receptor vertices.
        ligand_vertices (ndarray): Coordinates of the ligand vertices.
        receptor_point (ndarray): Coordinates of the receptor point.
        ligand_seed_point (ndarray): Coordinates of the ligand seed point.
        receptor_descriptor (ndarray): Descriptor vector of the receptor.
        ligand_descriptor (ndarray): Descriptor vector of the ligand.
        receptor_extreme (dict): Extreme values and points for receptor.
        ligand_extreme (dict): Extreme values and points for ligand.

    Returns:
        list[ndarray]: List of transformed ligand poses.
        list[ndarray]: List of combined transformation matrices used.
        ndarray: Translation vector used.
    """
    poses = []
    transformation_matrices = []
    receptor_min_eigenvector = receptor_descriptor['min_eigenvector']
    ligand_min_eigenvector = ligand_descriptor['min_eigenvector']

	
    # Step 1: Translation to align ligand seed point to receptor point
    translation_vector = receptor_point - ligand_seed_point
	#huyue 
    print("Debug : translation_vector len",len(translation_vector))
    print("translation_vector ",translation_vector)	
    print("DEBUG ligand_vertices ",ligand_vertices)
	#a_list = [item - 5 for item in a_list]
	#huyue
    ligand_translated = [item + translation_vector.T for item in ligand_vertices] 
    print("DEBUG ligand_translated ",ligand_translated)
	#huyue to (0,0,0)
    ligand_translated = [item - receptor_point.T for item in ligand_translated] 	
    print("DEBUG ligand_translated to 0,0,0",ligand_translated)	
    # Step 2: Rotation to align descriptors
    # The rotation axis is defined as the cross product of ligand_descriptor and receptor_descriptor
    rotation_axis = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis

    # Calculate the angle between the descriptors
    angle = np.arccos(np.clip(np.dot(ligand_min_eigenvector, receptor_min_eigenvector) / (np.linalg.norm(ligand_min_eigenvector) * np.linalg.norm(receptor_min_eigenvector)), -1.0, 1.0))
    rotation = R.from_rotvec(angle * rotation_axis)
    ligand_rotated = rotation.apply(ligand_translated)
    #print("DEBUG ligand_rotated ",ligand_rotated)	
    print("DEBUG ligand_rotated ",ligand_rotated)
    # Initialize combined rotation matrix with the descriptor alignment rotation
    combined_rotation_matrix = rotation.as_matrix()
    print("DEBUG combined_rotation_matrix ",combined_rotation_matrix)
    # Step 3: Align complementary extreme properties
    #for property_name in ['charge', 'hphob', 'hbond']:
    for property_name in ['charge']:
        receptor_max = receptor_extreme[property_name]['max_point']
        ligand_min = ligand_extreme[property_name]['min_point']

        # Define the rotation axis as the receptor descriptor, normalized
        rotation_axis_extreme = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)

        # Calculate vectors from the extreme points to the rotation axis (perpendicular vectors)
        vec_receptor_to_axis = receptor_max - receptor_point
        print("DEBUG Maybe erro vec_receptor_to_axis ",vec_receptor_to_axis)
        print("DEBUG Maybe erro ligand_min ",ligand_min)		
        # Transform ligand extreme point using previous transformations (translation + rotation)
        ligand_min_transformed = rotation.apply(ligand_min + translation_vector-receptor_point)
        print("DEBUG Maybe erro ligand_min ",ligand_min_transformed)		
        vec_ligand_to_axis = ligand_min_transformed
        print("DEBUG Maybe erro vec_ligand_to_axis ",vec_ligand_to_axis)
		
		
        # Project the vectors onto the plane orthogonal to the rotation axis to obtain the perpendicular component
        vec_receptor_proj = vec_receptor_to_axis - np.dot(vec_receptor_to_axis, rotation_axis_extreme) * rotation_axis_extreme
        vec_ligand_proj = vec_ligand_to_axis - np.dot(vec_ligand_to_axis, rotation_axis_extreme) * rotation_axis_extreme

        # Normalize the perpendicular vectors
        vec_receptor_proj /= np.linalg.norm(vec_receptor_proj)
        vec_ligand_proj /= np.linalg.norm(vec_ligand_proj)

        # Calculate the rotation angle between the two normalized perpendicular vectors
        rotation_angle_extreme = np.arccos(np.clip(np.dot( vec_ligand_proj, vec_receptor_proj), -1.0, 1.0))

        # Determine the direction of the rotation using the cross product
        cross_product = np.cross(vec_ligand_proj, vec_receptor_proj)
        if np.dot(cross_product, rotation_axis_extreme) < 0:
            rotation_angle_extreme = -rotation_angle_extreme

        # Apply the final rotation to align the extremes
        final_rotation = R.from_rotvec(rotation_angle_extreme * rotation_axis_extreme)
        final_pose = final_rotation.apply(ligand_rotated) + receptor_point
        print("DEBUG Maybe erro final_rotation ",final_rotation)
        # Update the combined rotation matrix
        #combined_rotation_matrix = final_rotation.as_matrix() @ combined_rotation_matrix
        print("DEBUG Maybe erro combined_rotation_matrix ",combined_rotation_matrix)
        # Store the resulting pose and transformation matrix
        poses.append(final_pose)
        transformation_matrices.append(combined_rotation_matrix)
    #return poses, transformation_matrices, translation_vector
    return poses, combined_rotation_matrix, translation_vector, receptor_point



def obtain_binding_poses(receptor_vertices, ligand_vertices, receptor_point, ligand_seed_point, receptor_descriptor, ligand_descriptor, receptor_extreme, ligand_extreme):
    """
    Calculate the binding poses for receptor and ligand.

    Parameters:
        receptor_vertices (ndarray): Coordinates of the receptor vertices.
        ligand_vertices (ndarray): Coordinates of the ligand vertices.
        receptor_point (ndarray): Coordinates of the receptor point.
        ligand_seed_point (ndarray): Coordinates of the ligand seed point.
        receptor_descriptor (ndarray): Descriptor vector of the receptor.
        ligand_descriptor (ndarray): Descriptor vector of the ligand.
        receptor_extreme (dict): Extreme values and points for receptor.
        ligand_extreme (dict): Extreme values and points for ligand.

    Returns:
        list[ndarray]: List of transformed ligand poses.
        list[ndarray]: List of combined transformation matrices used.
        ndarray: Translation vector used.
    """
    poses = []
    transformation_matrices = []
    receptor_min_eigenvector = receptor_descriptor['min_eigenvector']
    ligand_min_eigenvector = ligand_descriptor['min_eigenvector']
    receptor_min_eigenvector = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)
    ligand_min_eigenvector = ligand_min_eigenvector / np.linalg.norm(ligand_min_eigenvector)	
    # Calculate centroids
    receptor_centroid = np.mean(receptor_vertices, axis=0)
    ligand_centroid = np.mean(ligand_vertices, axis=0)

    # Step 1: Ensure receptor feature vectors point outward from centroids and ligand inward from centrods
    if np.dot(receptor_min_eigenvector, receptor_point - receptor_centroid) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector
    if np.dot(ligand_min_eigenvector, ligand_seed_point - ligand_centroid) > 0:
        ligand_min_eigenvector = -ligand_min_eigenvector
    #print("before ligand_vertices",ligand_vertices)
    # Step 2: Translation to align ligand seed point to receptor point
    translation_vector = receptor_point - ligand_seed_point
    ligand_translated = [item + translation_vector.T for item in ligand_vertices]
    ligand_translated = [item - receptor_point.T for item in ligand_translated]
    #print("after ligand_vertices",ligand_translated)
    # Step 3: Rotation to align descriptors
    if np.dot(receptor_min_eigenvector, ligand_min_eigenvector) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector	
    rotation_axis = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the angle between descriptors
    angle = np.arccos(np.clip(np.dot(ligand_min_eigenvector, receptor_min_eigenvector) / 
                              (np.linalg.norm(ligand_min_eigenvector) * np.linalg.norm(receptor_min_eigenvector)), -1.0, 1.0))
    cross_product_first = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    if np.dot(cross_product_first, rotation_axis) < 0:
        angle = -angle
							  
    rotation = R.from_rotvec(angle * rotation_axis)
    ligand_rotated = rotation.apply(ligand_translated)
    #print("totation ligand_vertices",ligand_rotated)
    # Initialize combined rotation matrix
    combined_rotation_matrix = rotation.as_matrix()

    # Step 4: Align complementary extreme properties
    for property_name in ['charge']:
        receptor_max = receptor_extreme[property_name]['max_point']
        ligand_min = ligand_extreme[property_name]['min_point']

        # Define rotation axis for extreme alignment
        rotation_axis_extreme = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)

        # Calculate vectors for extreme alignment
        vec_receptor_to_axis = receptor_max - receptor_point
        ligand_min_transformed = rotation.apply(ligand_min + translation_vector - receptor_point)
        vec_ligand_to_axis = ligand_min_transformed

        # Project the vectors onto the plane orthogonal to rotation axis
        vec_receptor_proj = vec_receptor_to_axis - np.dot(vec_receptor_to_axis, rotation_axis_extreme) * rotation_axis_extreme
        vec_ligand_proj = vec_ligand_to_axis - np.dot(vec_ligand_to_axis, rotation_axis_extreme) * rotation_axis_extreme

        # Normalize the vectors
        vec_receptor_proj /= np.linalg.norm(vec_receptor_proj)
        vec_ligand_proj /= np.linalg.norm(vec_ligand_proj)

        # Calculate rotation angle between vectors
        rotation_angle_extreme = np.arccos(np.clip(np.dot(vec_ligand_proj, vec_receptor_proj), -1.0, 1.0))

        # Determine rotation direction using cross product
        cross_product = np.cross(vec_ligand_proj, vec_receptor_proj)
        if np.dot(cross_product, rotation_axis_extreme) < 0:
            rotation_angle_extreme = -rotation_angle_extreme

        # Apply final rotation for extreme alignment
        final_rotation = R.from_rotvec(rotation_angle_extreme * rotation_axis_extreme)
        #print("final rotation ",final_rotation)		
        final_pose = final_rotation.apply(ligand_rotated) + receptor_point
        #print("final pose ",final_pose)		
        # Update rotation matrix and store pose
        combined_rotation_matrix = final_rotation.as_matrix() @ combined_rotation_matrix
        poses.append(final_pose)
        transformation_matrices.append(combined_rotation_matrix)
		#test	
        #t2=np.dot(ligand_translated, combined_rotation_matrix.T)+ receptor_point
        #print("t2 pose ",t2)	
        
		
		
    return poses, combined_rotation_matrix, translation_vector, receptor_point, ligand_seed_point

def obtain_binding_poses_charge_reverse(receptor_vertices, ligand_vertices, receptor_point, ligand_seed_point, receptor_descriptor, ligand_descriptor, receptor_extreme, ligand_extreme):
    """
    Calculate the binding poses for receptor and ligand.

    Parameters:
        receptor_vertices (ndarray): Coordinates of the receptor vertices.
        ligand_vertices (ndarray): Coordinates of the ligand vertices.
        receptor_point (ndarray): Coordinates of the receptor point.
        ligand_seed_point (ndarray): Coordinates of the ligand seed point.
        receptor_descriptor (ndarray): Descriptor vector of the receptor.
        ligand_descriptor (ndarray): Descriptor vector of the ligand.
        receptor_extreme (dict): Extreme values and points for receptor.
        ligand_extreme (dict): Extreme values and points for ligand.

    Returns:
        list[ndarray]: List of transformed ligand poses.
        list[ndarray]: List of combined transformation matrices used.
        ndarray: Translation vector used.
    """
    poses = []
    transformation_matrices = []
    receptor_min_eigenvector = receptor_descriptor['min_eigenvector']
    ligand_min_eigenvector = ligand_descriptor['min_eigenvector']
    receptor_min_eigenvector = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)
    ligand_min_eigenvector = ligand_min_eigenvector / np.linalg.norm(ligand_min_eigenvector)	
    # Calculate centroids
    receptor_centroid = np.mean(receptor_vertices, axis=0)
    ligand_centroid = np.mean(ligand_vertices, axis=0)

    # Step 1: Ensure receptor feature vectors point outward from centroids and ligand inward from centrods
    if np.dot(receptor_min_eigenvector, receptor_point - receptor_centroid) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector
    if np.dot(ligand_min_eigenvector, ligand_seed_point - ligand_centroid) > 0:
        ligand_min_eigenvector = -ligand_min_eigenvector
    #print("before ligand_vertices",ligand_vertices)
    # Step 2: Translation to align ligand seed point to receptor point
    translation_vector = receptor_point - ligand_seed_point
    ligand_translated = [item + translation_vector.T for item in ligand_vertices]
    ligand_translated = [item - receptor_point.T for item in ligand_translated]
    #print("after ligand_vertices",ligand_translated)
    # Step 3: Rotation to align descriptors
    if np.dot(receptor_min_eigenvector, ligand_min_eigenvector) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector	
    rotation_axis = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the angle between descriptors
    angle = np.arccos(np.clip(np.dot(ligand_min_eigenvector, receptor_min_eigenvector) / 
                              (np.linalg.norm(ligand_min_eigenvector) * np.linalg.norm(receptor_min_eigenvector)), -1.0, 1.0))
    cross_product_first = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    if np.dot(cross_product_first, rotation_axis) < 0:
        angle = -angle
							  
    rotation = R.from_rotvec(angle * rotation_axis)
    ligand_rotated = rotation.apply(ligand_translated)
    #print("totation ligand_vertices",ligand_rotated)
    # Initialize combined rotation matrix
    combined_rotation_matrix = rotation.as_matrix()

    # Step 4: Align complementary extreme properties
    for property_name in ['charge']:
        receptor_max = receptor_extreme[property_name]['min_point']
        ligand_min = ligand_extreme[property_name]['max_point']

        # Define rotation axis for extreme alignment
        rotation_axis_extreme = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)

        # Calculate vectors for extreme alignment
        vec_receptor_to_axis = receptor_max - receptor_point
        ligand_min_transformed = rotation.apply(ligand_min + translation_vector - receptor_point)
        vec_ligand_to_axis = ligand_min_transformed

        # Project the vectors onto the plane orthogonal to rotation axis
        vec_receptor_proj = vec_receptor_to_axis - np.dot(vec_receptor_to_axis, rotation_axis_extreme) * rotation_axis_extreme
        vec_ligand_proj = vec_ligand_to_axis - np.dot(vec_ligand_to_axis, rotation_axis_extreme) * rotation_axis_extreme

        # Normalize the vectors
        vec_receptor_proj /= np.linalg.norm(vec_receptor_proj)
        vec_ligand_proj /= np.linalg.norm(vec_ligand_proj)

        # Calculate rotation angle between vectors
        rotation_angle_extreme = np.arccos(np.clip(np.dot(vec_ligand_proj, vec_receptor_proj), -1.0, 1.0))

        # Determine rotation direction using cross product
        cross_product = np.cross(vec_ligand_proj, vec_receptor_proj)
        if np.dot(cross_product, rotation_axis_extreme) < 0:
            rotation_angle_extreme = -rotation_angle_extreme

        # Apply final rotation for extreme alignment
        final_rotation = R.from_rotvec(rotation_angle_extreme * rotation_axis_extreme)
        #print("final rotation ",final_rotation)		
        final_pose = final_rotation.apply(ligand_rotated) + receptor_point
        #print("final pose ",final_pose)		
        # Update rotation matrix and store pose
        combined_rotation_matrix = final_rotation.as_matrix() @ combined_rotation_matrix
        poses.append(final_pose)
        transformation_matrices.append(combined_rotation_matrix)
		#test	
        #t2=np.dot(ligand_translated, combined_rotation_matrix.T)+ receptor_point
        #print("t2 pose ",t2)	
        
		
		
    return poses, combined_rotation_matrix, translation_vector, receptor_point, ligand_seed_point

def obtain_binding_poses_hphob(receptor_vertices, ligand_vertices, receptor_point, ligand_seed_point, receptor_descriptor, ligand_descriptor, receptor_extreme, ligand_extreme):
    """
    Calculate the binding poses for receptor and ligand.

    Parameters:
        receptor_vertices (ndarray): Coordinates of the receptor vertices.
        ligand_vertices (ndarray): Coordinates of the ligand vertices.
        receptor_point (ndarray): Coordinates of the receptor point.
        ligand_seed_point (ndarray): Coordinates of the ligand seed point.
        receptor_descriptor (ndarray): Descriptor vector of the receptor.
        ligand_descriptor (ndarray): Descriptor vector of the ligand.
        receptor_extreme (dict): Extreme values and points for receptor.
        ligand_extreme (dict): Extreme values and points for ligand.

    Returns:
        list[ndarray]: List of transformed ligand poses.
        list[ndarray]: List of combined transformation matrices used.
        ndarray: Translation vector used.
    """
    poses = []
    transformation_matrices = []
    receptor_min_eigenvector = receptor_descriptor['min_eigenvector']
    ligand_min_eigenvector = ligand_descriptor['min_eigenvector']
    receptor_min_eigenvector = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)
    ligand_min_eigenvector = ligand_min_eigenvector / np.linalg.norm(ligand_min_eigenvector)	
    # Calculate centroids
    receptor_centroid = np.mean(receptor_vertices, axis=0)
    ligand_centroid = np.mean(ligand_vertices, axis=0)

    # Step 1: Ensure receptor feature vectors point outward from centroids and ligand inward from centrods
    if np.dot(receptor_min_eigenvector, receptor_point - receptor_centroid) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector
    if np.dot(ligand_min_eigenvector, ligand_seed_point - ligand_centroid) > 0:
        ligand_min_eigenvector = -ligand_min_eigenvector
    #print("before ligand_vertices",ligand_vertices)
    # Step 2: Translation to align ligand seed point to receptor point
    translation_vector = receptor_point - ligand_seed_point
    ligand_translated = [item + translation_vector.T for item in ligand_vertices]
    ligand_translated = [item - receptor_point.T for item in ligand_translated]
    #print("after ligand_vertices",ligand_translated)
    # Step 3: Rotation to align descriptors
    if np.dot(receptor_min_eigenvector, ligand_min_eigenvector) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector	
    rotation_axis = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the angle between descriptors
    angle = np.arccos(np.clip(np.dot(ligand_min_eigenvector, receptor_min_eigenvector) / 
                              (np.linalg.norm(ligand_min_eigenvector) * np.linalg.norm(receptor_min_eigenvector)), -1.0, 1.0))
    cross_product_first = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    if np.dot(cross_product_first, rotation_axis) < 0:
        angle = -angle
							  
    rotation = R.from_rotvec(angle * rotation_axis)
    ligand_rotated = rotation.apply(ligand_translated)
    #print("totation ligand_vertices",ligand_rotated)
    # Initialize combined rotation matrix
    combined_rotation_matrix = rotation.as_matrix()

    # Step 4: Align complementary extreme properties
    for property_name in ['hphob']:
        receptor_max = receptor_extreme[property_name]['max_point']
        ligand_min = ligand_extreme[property_name]['min_point']

        # Define rotation axis for extreme alignment
        rotation_axis_extreme = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)

        # Calculate vectors for extreme alignment
        vec_receptor_to_axis = receptor_max - receptor_point
        ligand_min_transformed = rotation.apply(ligand_min + translation_vector - receptor_point)
        vec_ligand_to_axis = ligand_min_transformed

        # Project the vectors onto the plane orthogonal to rotation axis
        vec_receptor_proj = vec_receptor_to_axis - np.dot(vec_receptor_to_axis, rotation_axis_extreme) * rotation_axis_extreme
        vec_ligand_proj = vec_ligand_to_axis - np.dot(vec_ligand_to_axis, rotation_axis_extreme) * rotation_axis_extreme

        # Normalize the vectors
        vec_receptor_proj /= np.linalg.norm(vec_receptor_proj)
        vec_ligand_proj /= np.linalg.norm(vec_ligand_proj)

        # Calculate rotation angle between vectors
        rotation_angle_extreme = np.arccos(np.clip(np.dot(vec_ligand_proj, vec_receptor_proj), -1.0, 1.0))

        # Determine rotation direction using cross product
        cross_product = np.cross(vec_ligand_proj, vec_receptor_proj)
        if np.dot(cross_product, rotation_axis_extreme) < 0:
            rotation_angle_extreme = -rotation_angle_extreme

        # Apply final rotation for extreme alignment
        final_rotation = R.from_rotvec(rotation_angle_extreme * rotation_axis_extreme)
        #print("final rotation ",final_rotation)		
        final_pose = final_rotation.apply(ligand_rotated) + receptor_point
        #print("final pose ",final_pose)		
        # Update rotation matrix and store pose
        combined_rotation_matrix = final_rotation.as_matrix() @ combined_rotation_matrix
        poses.append(final_pose)
        transformation_matrices.append(combined_rotation_matrix)
		#test	
        #t2=np.dot(ligand_translated, combined_rotation_matrix.T)+ receptor_point
        #print("t2 pose ",t2)	
        
		
		
    return poses, combined_rotation_matrix, translation_vector, receptor_point, ligand_seed_point	

def obtain_binding_poses_hphob_reverse(receptor_vertices, ligand_vertices, receptor_point, ligand_seed_point, receptor_descriptor, ligand_descriptor, receptor_extreme, ligand_extreme):
    """
    Calculate the binding poses for receptor and ligand.

    Parameters:
        receptor_vertices (ndarray): Coordinates of the receptor vertices.
        ligand_vertices (ndarray): Coordinates of the ligand vertices.
        receptor_point (ndarray): Coordinates of the receptor point.
        ligand_seed_point (ndarray): Coordinates of the ligand seed point.
        receptor_descriptor (ndarray): Descriptor vector of the receptor.
        ligand_descriptor (ndarray): Descriptor vector of the ligand.
        receptor_extreme (dict): Extreme values and points for receptor.
        ligand_extreme (dict): Extreme values and points for ligand.

    Returns:
        list[ndarray]: List of transformed ligand poses.
        list[ndarray]: List of combined transformation matrices used.
        ndarray: Translation vector used.
    """
    poses = []
    transformation_matrices = []
    receptor_min_eigenvector = receptor_descriptor['min_eigenvector']
    ligand_min_eigenvector = ligand_descriptor['min_eigenvector']
    receptor_min_eigenvector = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)
    ligand_min_eigenvector = ligand_min_eigenvector / np.linalg.norm(ligand_min_eigenvector)	
    # Calculate centroids
    receptor_centroid = np.mean(receptor_vertices, axis=0)
    ligand_centroid = np.mean(ligand_vertices, axis=0)

    # Step 1: Ensure receptor feature vectors point outward from centroids and ligand inward from centrods
    if np.dot(receptor_min_eigenvector, receptor_point - receptor_centroid) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector
    if np.dot(ligand_min_eigenvector, ligand_seed_point - ligand_centroid) > 0:
        ligand_min_eigenvector = -ligand_min_eigenvector
    #print("before ligand_vertices",ligand_vertices)
    # Step 2: Translation to align ligand seed point to receptor point
    translation_vector = receptor_point - ligand_seed_point
    ligand_translated = [item + translation_vector.T for item in ligand_vertices]
    ligand_translated = [item - receptor_point.T for item in ligand_translated]
    #print("after ligand_vertices",ligand_translated)
    # Step 3: Rotation to align descriptors
    if np.dot(receptor_min_eigenvector, ligand_min_eigenvector) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector	
    rotation_axis = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Calculate the angle between descriptors
    angle = np.arccos(np.clip(np.dot(ligand_min_eigenvector, receptor_min_eigenvector) / 
                              (np.linalg.norm(ligand_min_eigenvector) * np.linalg.norm(receptor_min_eigenvector)), -1.0, 1.0))
    cross_product_first = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    if np.dot(cross_product_first, rotation_axis) < 0:
        angle = -angle
							  
    rotation = R.from_rotvec(angle * rotation_axis)
    ligand_rotated = rotation.apply(ligand_translated)
    #print("totation ligand_vertices",ligand_rotated)
    # Initialize combined rotation matrix
    combined_rotation_matrix = rotation.as_matrix()

    # Step 4: Align complementary extreme properties
    for property_name in ['hphob']:
        receptor_max = receptor_extreme[property_name]['min_point']
        ligand_min = ligand_extreme[property_name]['max_point']

        # Define rotation axis for extreme alignment
        rotation_axis_extreme = receptor_min_eigenvector / np.linalg.norm(receptor_min_eigenvector)

        # Calculate vectors for extreme alignment
        vec_receptor_to_axis = receptor_max - receptor_point
        ligand_min_transformed = rotation.apply(ligand_min + translation_vector - receptor_point)
        vec_ligand_to_axis = ligand_min_transformed

        # Project the vectors onto the plane orthogonal to rotation axis
        vec_receptor_proj = vec_receptor_to_axis - np.dot(vec_receptor_to_axis, rotation_axis_extreme) * rotation_axis_extreme
        vec_ligand_proj = vec_ligand_to_axis - np.dot(vec_ligand_to_axis, rotation_axis_extreme) * rotation_axis_extreme

        # Normalize the vectors
        vec_receptor_proj /= np.linalg.norm(vec_receptor_proj)
        vec_ligand_proj /= np.linalg.norm(vec_ligand_proj)

        # Calculate rotation angle between vectors
        rotation_angle_extreme = np.arccos(np.clip(np.dot(vec_ligand_proj, vec_receptor_proj), -1.0, 1.0))

        # Determine rotation direction using cross product
        cross_product = np.cross(vec_ligand_proj, vec_receptor_proj)
        if np.dot(cross_product, rotation_axis_extreme) < 0:
            rotation_angle_extreme = -rotation_angle_extreme

        # Apply final rotation for extreme alignment
        final_rotation = R.from_rotvec(rotation_angle_extreme * rotation_axis_extreme)
        #print("final rotation ",final_rotation)		
        final_pose = final_rotation.apply(ligand_rotated) + receptor_point
        #print("final pose ",final_pose)		
        # Update rotation matrix and store pose
        combined_rotation_matrix = final_rotation.as_matrix() @ combined_rotation_matrix
        poses.append(final_pose)
        transformation_matrices.append(combined_rotation_matrix)
		#test	
        #t2=np.dot(ligand_translated, combined_rotation_matrix.T)+ receptor_point
        #print("t2 pose ",t2)	
        
		
		
    return poses, combined_rotation_matrix, translation_vector, receptor_point, ligand_seed_point	

from scipy.spatial import KDTree
import numpy as np
from scipy.spatial.transform import Rotation as R

def obtain_binding_poses_hessian_f1(receptor_vertices, ligand_vertices, receptor_point, ligand_seed_point, receptor_descriptor, ligand_descriptor):
    """
    Calculate the binding poses for receptor and ligand based on Hessian matrix descriptors.

    Parameters:
        receptor_vertices (ndarray): Coordinates of the receptor vertices.
        ligand_vertices (ndarray): Coordinates of the ligand vertices.
        receptor_point (ndarray): Coordinates of the receptor point.
        ligand_seed_point (ndarray): Coordinates of the ligand seed point.
        receptor_descriptor (ndarray): Sorted eigenvectors of the receptor (from smallest to largest eigenvalue).
        ligand_descriptor (ndarray): Sorted eigenvectors of the ligand (from smallest to largest eigenvalue).

    Returns:
        list[ndarray]: List of transformed ligand poses.
        list[ndarray]: List of combined transformation matrices used.
        ndarray: Translation vector used.
    """
    poses = []
    transformation_matrices = []

    # Use the smallest eigenvector (min curvature direction) for initial alignment
    receptor_min_eigenvector = receptor_descriptor[0]
    ligand_min_eigenvector = ligand_descriptor[0]
    
    # Normalize these eigenvectors
    receptor_min_eigenvector /= np.linalg.norm(receptor_min_eigenvector)
    ligand_min_eigenvector /= np.linalg.norm(ligand_min_eigenvector)
    
    # Calculate centroids for initial orientation check
    receptor_centroid = np.mean(receptor_vertices, axis=0)
    ligand_centroid = np.mean(ligand_vertices, axis=0)

    # Ensure eigenvectors point outward/inward as required
    if np.dot(receptor_min_eigenvector, receptor_point - receptor_centroid) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector
    if np.dot(ligand_min_eigenvector, ligand_seed_point - ligand_centroid) > 0:
        ligand_min_eigenvector = -ligand_min_eigenvector

    # Step 1: Translation to align ligand seed point to receptor point
    translation_vector = receptor_point - ligand_seed_point
    ligand_translated = ligand_vertices + translation_vector

    # Step 2: Rotation to align primary descriptors (smallest eigenvectors)
    rotation_axis = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = np.arccos(np.clip(np.dot(ligand_min_eigenvector, receptor_min_eigenvector), -1.0, 1.0))
    rotation = R.from_rotvec(angle * rotation_axis)
    ligand_rotated = rotation.apply(ligand_translated)

    combined_rotation_matrix = rotation.as_matrix()

    # Step 3: Generate additional orientations using remaining eigenvectors
    receptor_eigenvector_other1 = receptor_descriptor[1]
    receptor_eigenvector_other2 = receptor_descriptor[2]
    ligand_max_eigenvector = ligand_descriptor[2] / np.linalg.norm(ligand_descriptor[2])

    for receptor_eigenvector in [receptor_eigenvector_other1, receptor_eigenvector_other2]:
        for sign1 in [1, -1]:  # Iterate over positive and negative directions of ligand eigenvector
            for sign2 in [1, -1]:  # Iterate over positive and negative directions of receptor eigenvector
                ligand_vector_adjusted = sign1 * ligand_max_eigenvector
                receptor_vector_adjusted = sign2 * receptor_eigenvector / np.linalg.norm(receptor_eigenvector)
                
                # Calculate rotation axis and angle for secondary alignment
                rotation_axis_secondary = np.cross(ligand_vector_adjusted, receptor_vector_adjusted)
                if np.linalg.norm(rotation_axis_secondary) < 1e-6:
                    continue  # Skip if vectors are nearly parallel
                rotation_axis_secondary /= np.linalg.norm(rotation_axis_secondary)
                
                # Calculate angle between ligand and receptor vectors
                angle_secondary = np.arccos(np.clip(np.dot(ligand_vector_adjusted, receptor_vector_adjusted), -1.0, 1.0))
                rotation_secondary = R.from_rotvec(angle_secondary * rotation_axis_secondary)

                # Apply secondary rotation to ligand
                final_pose = rotation_secondary.apply(ligand_rotated) + receptor_point
                final_combined_rotation_matrix = rotation_secondary.as_matrix() @ combined_rotation_matrix

                # Store final pose and transformation matrix
                poses.append(final_pose)
                transformation_matrices.append(final_combined_rotation_matrix)

    return poses, transformation_matrices, translation_vector, receptor_point, ligand_seed_point



from scipy.spatial import KDTree
import numpy as np
from scipy.spatial.transform import Rotation as R

def obtain_binding_poses_hessian(receptor_vertices, ligand_vertices, receptor_point, ligand_seed_point, receptor_descriptor, ligand_descriptor):
    """
    Calculate the binding poses for receptor and ligand based on Hessian matrix descriptors.

    Parameters:
        receptor_vertices (ndarray): Coordinates of the receptor vertices.
        ligand_vertices (ndarray): Coordinates of the ligand vertices.
        receptor_point (ndarray): Coordinates of the receptor point.
        ligand_seed_point (ndarray): Coordinates of the ligand seed point.
        receptor_descriptor (ndarray): Sorted eigenvectors of the receptor (from smallest to largest eigenvalue).
        ligand_descriptor (ndarray): Sorted eigenvectors of the ligand (from smallest to largest eigenvalue).

    Returns:
        list[ndarray]: List of transformed ligand poses.
        list[ndarray]: List of combined transformation matrices used.
        ndarray: Translation vector used.
    """
    poses = []
    transformation_matrices = []

    # Use the smallest eigenvector (min curvature direction) for initial alignment
    receptor_min_eigenvector = receptor_descriptor[0]
    ligand_min_eigenvector = ligand_descriptor[0]
    
    # Normalize these eigenvectors
    receptor_min_eigenvector /= np.linalg.norm(receptor_min_eigenvector)
    ligand_min_eigenvector /= np.linalg.norm(ligand_min_eigenvector)
    
    # Calculate centroids for initial orientation check
    receptor_centroid = np.mean(receptor_vertices, axis=0)
    ligand_centroid = np.mean(ligand_vertices, axis=0)

    # Ensure eigenvectors point outward/inward as required
    if np.dot(receptor_min_eigenvector, receptor_point - receptor_centroid) < 0:
        receptor_min_eigenvector = -receptor_min_eigenvector
    if np.dot(ligand_min_eigenvector, ligand_seed_point - ligand_centroid) > 0:
        ligand_min_eigenvector = -ligand_min_eigenvector

    # Step 1: Translation to align ligand seed point to receptor point
    translation_vector = receptor_point - ligand_seed_point
    ligand_translated = ligand_vertices + translation_vector

    # Step 2: Rotation to align primary descriptors (smallest eigenvectors)
    rotation_axis = np.cross(ligand_min_eigenvector, receptor_min_eigenvector)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = np.arccos(np.clip(np.dot(ligand_min_eigenvector, receptor_min_eigenvector), -1.0, 1.0))
    rotation = R.from_rotvec(angle * rotation_axis)
    ligand_rotated = rotation.apply(ligand_translated)

    combined_rotation_matrix = rotation.as_matrix()

    # Step 3: Generate additional orientations using remaining eigenvectors
    receptor_eigenvector_other1 = receptor_descriptor[1]
    receptor_eigenvector_other2 = receptor_descriptor[2]
    
    # Set ligand_max_eigenvector once, since it will remain constant across iterations
    ligand_vector_adjusted = ligand_descriptor[2] / np.linalg.norm(ligand_descriptor[2])

    for receptor_eigenvector in [receptor_eigenvector_other1, receptor_eigenvector_other2]:
        for sign2 in [1, -1]:  # Iterate over positive and negative directions of receptor eigenvector
            receptor_vector_adjusted = sign2 * receptor_eigenvector / np.linalg.norm(receptor_eigenvector)
                
            # Calculate rotation axis and angle for secondary alignment
            rotation_axis_secondary = np.cross(ligand_vector_adjusted, receptor_vector_adjusted)
            if np.linalg.norm(rotation_axis_secondary) < 1e-6:
                continue  # Skip if vectors are nearly parallel
            rotation_axis_secondary /= np.linalg.norm(rotation_axis_secondary)
                
            # Calculate angle between ligand and receptor vectors
            angle_secondary = np.arccos(np.clip(np.dot(ligand_vector_adjusted, receptor_vector_adjusted), -1.0, 1.0))
            rotation_secondary = R.from_rotvec(angle_secondary * rotation_axis_secondary)

            # Apply secondary rotation to ligand
            final_pose = rotation_secondary.apply(ligand_rotated) + receptor_point
            final_combined_rotation_matrix = rotation_secondary.as_matrix() @ combined_rotation_matrix

            # Store final pose and transformation matrix
            poses.append(final_pose)
            transformation_matrices.append(final_combined_rotation_matrix)

    return poses, transformation_matrices, translation_vector, receptor_point, ligand_seed_point






	
# Align surface patches for pose generation
def align_patches(seed_points_receptor, seed_points_ligand):
    poses = []
    for seed_receptor in seed_points_receptor:
        for seed_ligand in seed_points_ligand:
            # Calculate translation vector
            translation = seed_receptor - seed_ligand

            # Calculate rotation matrix using a placeholder method (Formula 4-7)
            rotation_matrix = np.eye(3)  # Placeholder for actual rotation calculation

            # Generate a pose by applying rotation and translation to the ligand seed
            rotated_pose = np.dot(rotation_matrix, seed_ligand) + translation
            poses.append((rotation_matrix, translation))
    return poses

# Function to load a PDB file
def load_pdb(filename):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(filename), filename)
    return structure

# Function to save the pose as a PDB file
def save_pdb(structure, rotation_matrix, translation_vector, filename):
    moved_structure = structure.copy()
    for atom in moved_structure.get_atoms():
        # Apply rotation and translation
        atom_coord = np.dot(rotation_matrix, atom.coord) + translation_vector
        atom.set_coord(atom_coord)
    
    io = PDBIO()
    io.set_structure(moved_structure)
    io.save(filename)

	
# Function to score the poses based on Hbond, Charge, and Hydrophobicity
def score_poses_with_seed_interaction(receptor_mesh, ligand_mesh, rotation_matrices, translation_vectors, receptor_features, ligand_features, seed_points, ligand_seed_points, radius=5.0):
    """
    Score the poses based on electrostatic, hydrogen bond, and hydrophobic interactions within a certain radius of each receptor seed point.

    Parameters:
        receptor_mesh: ndarray
            The receptor mesh coordinates.
        ligand_mesh: ndarray
            The ligand mesh coordinates.
        rotation_matrices: list of ndarray
            List of rotation matrices for each pose.
        translation_vectors: list of ndarray
            List of translation vectors for each pose.
        receptor_features: dict
            The features of the receptor including charge, etc.
        ligand_features: dict
            The features of the ligand including charge, etc.
        seed_points: list of ndarray
            List of coordinates of seed points on the receptor.
        ligand_seed_points: list of ndarray, optional
            List of coordinates of seed points on the ligand. If provided, should match length of seed_points.
        radius: float
            The radius within which to consider receptor-ligand interactions.

    Returns:
        top_poses: list of tuples
            The top scoring poses including rotation and translation matrices.
        top_scores: list of floats
            The scores of the top poses.
        top_receptor_seed_points: list of ndarray
            Corresponding receptor seed points for the top poses.
    """
    scores = []
    best_poses = []
    best_receptor_seed_points = []

    # Create KDTree for fast neighbor search on receptor mesh
    kdtree = KDTree(receptor_mesh)

    # Loop over each seed point and its corresponding transformation
    for idx, (seed_point, rotation_matrix, translation_vector) in enumerate(zip(seed_points, rotation_matrices, translation_vectors)):
        # Find neighbors within the specified radius of the current seed point
        neighbors_idx = kdtree.query_ball_point(seed_point, r=radius)

        # Check if there are neighbors in the radius
        if len(neighbors_idx) == 0:
            print(f"No neighbors found within radius for seed point {idx}")
            scores.append(float('inf'))  # Assign a high score if no neighbors are found
            best_poses.append((rotation_matrix, translation_vector))
            best_receptor_seed_points.append(seed_point)			
            continue

        # Get receptor features and positions near the current seed point
        receptor_charges = receptor_features['charge'][neighbors_idx]
        receptor_hbonds = receptor_features['hbond'][neighbors_idx]
        receptor_hphob = receptor_features['hphob'][neighbors_idx]
        #receptor_positions = receptor_mesh[neighbors_idx] - seed_point.T
        receptor_positions = receptor_mesh[neighbors_idx]
        # Apply the transformation to ligand mesh to get new pose
        ligand_transformed = np.dot((ligand_mesh + translation_vector.T - seed_point.T), rotation_matrix.T)
		
        # Get ligand features and positions for the transformed pose
        ligand_positions = ligand_transformed + seed_point.T
        #print("ligand postions in score :", ligand_positions)
        ligand_charges = ligand_features['charge']
        ligand_hbonds = ligand_features['hbond']
        ligand_hphob = ligand_features['hphob']

        # Compute the interaction scores for charge, hydrogen bond, and hydrophobicity
        charge_score = calculate_charge_score(receptor_positions, receptor_charges, ligand_positions, ligand_charges)
        hbond_score = calculate_hbond_score(receptor_positions, receptor_hbonds, ligand_positions, ligand_hbonds)
        hydrophobicity_score = calculate_hydrophobicity_score(receptor_positions, receptor_hphob, ligand_positions, ligand_hphob)
        #clash_score = calculate_clash_score(receptor_positions, ligand_positions)
        # Combine scores into a total score (weights can be adjusted based on importance)
        #total_score = charge_score + hbond_score + hydrophobicity_score + clash_score
        total_score = charge_score + hbond_score + hydrophobicity_score 		
        #print("number, total score : ",idx,total_score)
        scores.append(total_score)
        best_poses.append((rotation_matrix, translation_vector))
        best_receptor_seed_points.append(seed_point)	

    # Get the top two poses with the lowest scores
    sorted_indices = np.argsort(scores)[:200]
    top_poses = [best_poses[i] for i in sorted_indices]
    top_scores = [scores[i] for i in sorted_indices]
    top_receptor_seed_points = [best_receptor_seed_points[i] for i in sorted_indices]
	
    return top_poses, top_scores, top_receptor_seed_points
	
	
# Function to score the poses based on Hbond, Charge, and Hydrophobicity
def score_poses_with_seed_interaction_f4(receptor_mesh, ligand_mesh, rotation_matrices, translation_vectors, receptor_features, ligand_features, seed_points, ligand_seed_points, radius=5.0):
    """
    Score the poses based on simple electrostatic interactions within a certain radius of each receptor seed point.

    Parameters:
        receptor_mesh: ndarray
            The receptor mesh coordinates.
        ligand_mesh: ndarray
            The ligand mesh coordinates.
        rotation_matrices: list of ndarray
            List of rotation matrices for each pose.
        translation_vectors: list of ndarray
            List of translation vectors for each pose.
        receptor_features: dict
            The features of the receptor including charge, etc.
        ligand_features: dict
            The features of the ligand including charge, etc.
        seed_points: list of ndarray
            List of coordinates of seed points on the receptor.
        ligand_seed_points: list of ndarray, optional
            List of coordinates of seed points on the ligand. If provided, should match length of seed_points.
        radius: float
            The radius within which to consider receptor-ligand interactions.

    Returns:
        top_poses: list of tuples
            The top scoring poses including rotation and translation matrices.
        top_scores: list of floats
            The scores of the top poses.
        top_ligand_seed_points: list of ndarray
            Corresponding ligand seed points for the top poses.
    """
    scores = []
    best_poses = []
    best_ligand_seed_points = []
    best_receptor_seed_points= []
    NullNumber=ligand_seed_points
    # Create KDTree for fast neighbor search on receptor mesh
    kdtree = KDTree(receptor_mesh)

    # Loop over each seed point and its corresponding transformation
    for idx, (seed_point, rotation_matrix, translation_vector) in enumerate(zip(seed_points, rotation_matrices, translation_vectors)):
        # Find neighbors within the specified radius of the current seed point
        neighbors_idx = kdtree.query_ball_point(seed_point, r=radius)

        # Check if there are neighbors in the radius
        if len(neighbors_idx) == 0:
            print(f"No neighbors found within radius for seed point {idx}")
            scores.append(float('-inf'))  # Assign a very low score if no neighbors are found
            best_poses.append((rotation_matrix, translation_vector))
            best_receptor_seed_points.append(seed_point)			
            #best_ligand_seed_points.append(ligand_seed_points[idx] if ligand_seed_points is not None else None)
            continue

        # Get receptor charges and positions near the current seed point
        receptor_charges = receptor_features['charge'][neighbors_idx]
        receptor_positions = receptor_mesh[neighbors_idx]

        # Apply the transformation to ligand mesh to get new pose
        ligand_transformed = np.dot((ligand_mesh+ translation_vector), rotation_matrix.T) 
		
        # Get ligand charges and positions for the transformed pose
        ligand_positions = ligand_transformed
        ligand_charges = ligand_features['charge']

        # Compute the charge-based interaction score
        charge_score = calculate_charge_score(receptor_positions, receptor_charges, ligand_positions, ligand_charges)
        scores.append(charge_score)
        best_poses.append((rotation_matrix, translation_vector))
        best_receptor_seed_points.append(seed_point)	
        #best_ligand_seed_points.append(ligand_seed_points[idx] if ligand_seed_points is not None else None)

    # Get the top two poses with the best scores
    #sorted_indices = np.argsort(scores)[-2:][::-1]
    # 从低到高排序并选择得分最低的两个
    sorted_indices = np.argsort(scores)[:500]
	
    top_poses = [best_poses[i] for i in sorted_indices]
    top_scores = [scores[i] for i in sorted_indices]
    #top_ligand_seed_points = [best_ligand_seed_points[i] for i in sorted_indices]
    top_receptor_seed_points = [best_receptor_seed_points[i] for i in sorted_indices]
	
    return top_poses, top_scores, top_receptor_seed_points


def score_poses_with_seed_interaction_f3(receptor_mesh, ligand_mesh, rotation_matrices, translation_vectors, receptor_features, ligand_features, seed_point, radius=5.0):
    """
    Score the poses based only on simple electrostatic interactions within a certain radius of a receptor seed point.

    Parameters:
        receptor_mesh: ndarray
            The receptor mesh coordinates.
        ligand_mesh: ndarray
            The ligand mesh coordinates.
        rotation_matrices: list of ndarray
            List of rotation matrices for each pose.
        translation_vectors: list of ndarray
            List of translation vectors for each pose.
        receptor_features: dict
            The features of the receptor including charge, etc.
        ligand_features: dict
            The features of the ligand including charge, etc.
        seed_point: ndarray
            The coordinates of the seed point on the receptor.
        radius: float
            The radius within which to consider receptor-ligand interactions.

    Returns:
        best_poses: list of tuples
            The top scoring poses.
        best_scores: list of floats
            The scores of the top poses.
    """
    scores = []
    kdtree = KDTree(receptor_mesh)  # Create a KDTree for fast neighbor search
    neighbors_idx = kdtree.query_ball_point(seed_point, r=radius)
    print("neighbors_idx line 436 :", neighbors_idx)
	
	# Flatten the list of lists into a single list of indices
    #neighbors_idx_flattened = [item for sublist in neighbors_idx for item in sublist]
    
	#for neighbors_idx_flattened in neighbors_idx:
    # Convert the flattened list to a numpy array of integers
	#	neighbors_idx_array = np.array(neighbors_idx_flattened, dtype=int)


    receptor_charges = receptor_features['charge'][neighbors_idx]  # Get receptor charges near the seed point
    receptor_positions = receptor_mesh[neighbors_idx]

    for idx, (rotation_matrix, translation_vector) in enumerate(zip(rotation_matrices, translation_vectors)):
        
		#if isinstance(ligand_mesh, list):
		#	ligand_mesh = np.array(ligand_mesh)
        if isinstance(rotation_matrix, list):
            rotation_matrix = np.array(rotation_matrix)
        if isinstance(translation_vector, list):
            translation_vector = np.array(translation_vector)

		#huyue	
		# Apply transformation to ligand to get new pose
        #moved_ligand = np.dot(ligand_mesh, rotation_matrix.T) + translation_vector
        #R.from_matrix(rotation_matrix)
        ligand_mesh=[item + translation_vector.T for item in ligand_mesh] 
        moved_ligand = np.dot(ligand_mesh, rotation_matrix.T)
        # Extract ligand positions and charges
        ligand_positions = moved_ligand
        ligand_charges = ligand_features['charge']

        # Compute the charge-based interaction score
        charge_score = calculate_charge_score(receptor_positions, receptor_charges, ligand_positions, ligand_charges)
        scores.append(charge_score)

    # Get the top two poses with the best scores
    sorted_indices = np.argsort(scores)[-2:][::-1]
    best_poses = [(rotation_matrices[i], translation_vectors[i]) for i in sorted_indices]
    best_scores = [scores[i] for i in sorted_indices]

    return best_poses, best_scores

def calculate_charge_score_f15(receptor_positions, receptor_charges, ligand_positions, ligand_charges):
    """
    Calculate the electrostatic interaction score between receptor and ligand charges.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_charges: ndarray
            Charges of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_charges: ndarray
            Charges of ligand atoms.

    Returns:
        float: The total electrostatic interaction score.
    """
    score = 0.0
    for r_pos, r_charge in zip(receptor_positions, receptor_charges):
        for l_pos, l_charge in zip(ligand_positions, ligand_charges):
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0:
                score += (r_charge * l_charge) / distance  # Coulomb's law
    return score

import numpy as np

def calculate_charge_score(receptor_positions, receptor_charges, ligand_positions, ligand_charges):
    """
    Calculate the electrostatic interaction score between receptor and ligand charges.
    Only the nearest ligand for each receptor is considered.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_charges: ndarray
            Charges of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_charges: ndarray
            Charges of ligand atoms.

    Returns:
        float: The total electrostatic interaction score.
    """
    score = 0.0
    for r_pos, r_charge in zip(receptor_positions, receptor_charges):
        # Initialize the nearest distance and corresponding ligand charge
        min_distance = float('inf')
        nearest_charge = 0.0
        
        for l_pos, l_charge in zip(ligand_positions, ligand_charges):
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0 and distance < min_distance:
                min_distance = distance
                nearest_charge = l_charge
        
        if min_distance < float('inf'):  # Ensure we found a nearest ligand
            score += (r_charge * nearest_charge) / min_distance  # Coulomb's law

    return score
	
	
# Function to calculate hydrogen bond interaction score
def calculate_hbond_score_f15(receptor_positions, receptor_hbonds, ligand_positions, ligand_hbonds):
    """
    Calculate the hydrogen bond interaction score between receptor and ligand based on distance.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_hbonds: ndarray
            Hydrogen bond properties of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_hbonds: ndarray
            Hydrogen bond properties of ligand atoms.

    Returns:
        float: The total hydrogen bond interaction score.
    """
    score = 0.0
    for r_pos, r_hbond in zip(receptor_positions, receptor_hbonds):
        for l_pos, l_hbond in zip(ligand_positions, ligand_hbonds):
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0:
                score += (r_hbond * l_hbond) / distance  # Distance-weighted interaction
    return score
import numpy as np

def calculate_hbond_score(receptor_positions, receptor_hbonds, ligand_positions, ligand_hbonds):
    """
    Calculate the hydrogen bond interaction score between receptor and ligand based on distance.
    Only the nearest ligand for each receptor is considered.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_hbonds: ndarray
            Hydrogen bond properties of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_hbonds: ndarray
            Hydrogen bond properties of ligand atoms.

    Returns:
        float: The total hydrogen bond interaction score.
    """
    score = 0.0
    for r_pos, r_hbond in zip(receptor_positions, receptor_hbonds):
        # Initialize the nearest distance and corresponding ligand hydrogen bond
        min_distance = float('inf')
        nearest_hbond = 0.0
        
        for l_pos, l_hbond in zip(ligand_positions, ligand_hbonds):
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0 and distance < min_distance:
                min_distance = distance
                nearest_hbond = l_hbond
        
        if min_distance < float('inf'):  # Ensure we found a nearest ligand
            score += (r_hbond * nearest_hbond) / min_distance  # Distance-weighted interaction

    return score


# Function to calculate hydrophobic interaction score
def calculate_hydrophobicity_score_f15(receptor_positions, receptor_hphob, ligand_positions, ligand_hphob):
    """
    Calculate the hydrophobic interaction score between receptor and ligand based on distance.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_hphob: ndarray
            Hydrophobic properties of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_hphob: ndarray
            Hydrophobic properties of ligand atoms.

    Returns:
        float: The total hydrophobic interaction score.
    """
    score = 0.0
    for r_pos, r_hphob in zip(receptor_positions, receptor_hphob):
        for l_pos, l_hphob in zip(ligand_positions, ligand_hphob):
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0:
                score += (r_hphob * l_hphob) / distance  # Distance-weighted interaction
    return score
import numpy as np

def calculate_hydrophobicity_score(receptor_positions, receptor_hphob, ligand_positions, ligand_hphob):
    """
    Calculate the hydrophobic interaction score between receptor and ligand based on distance.
    Only the nearest ligand for each receptor is considered.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_hphob: ndarray
            Hydrophobic properties of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_hphob: ndarray
            Hydrophobic properties of ligand atoms.

    Returns:
        float: The total hydrophobic interaction score.
    """
    score = 0.0
    for r_pos, r_hphob in zip(receptor_positions, receptor_hphob):
        # Initialize the nearest distance and corresponding ligand hydrophobic property
        min_distance = float('inf')
        nearest_hphob = 0.0
        
        for l_pos, l_hphob in zip(ligand_positions, ligand_hphob):
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0 and distance < min_distance:
                min_distance = distance
                nearest_hphob = l_hphob
        
        if min_distance < float('inf'):  # Ensure we found a nearest ligand
            score += (r_hphob * nearest_hphob) / min_distance  # Distance-weighted interaction

    return score

# Function to calculate hydrophobic interaction score
def calculate_clash_score(receptor_positions, ligand_positions):
    """
    Calculate the hydrophobic interaction score between receptor and ligand based on distance.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_hphob: ndarray
            Hydrophobic properties of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_hphob: ndarray
            Hydrophobic properties of ligand atoms.

    Returns:
        float: The total hydrophobic interaction score.
    """
    score = 0.0
    for r_pos in receptor_positions:
        for l_pos in ligand_positions:
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0:
                score =score + (3.5 / distance) ** 12 - (3.5 / distance) ** 6 # Distance-weighted interaction
    return score
# Function to calculate hydrophobic interaction score
def calculate_clash_score_energy(receptor_positions, ligand_positions):
    """
    Calculate the hydrophobic interaction score between receptor and ligand based on distance.

    Parameters:
        receptor_positions: ndarray
            Coordinates of receptor atoms within the interaction radius.
        receptor_hphob: ndarray
            Hydrophobic properties of receptor atoms.
        ligand_positions: ndarray
            Coordinates of ligand atoms.
        ligand_hphob: ndarray
            Hydrophobic properties of ligand atoms.

    Returns:
        float: The total hydrophobic interaction score.
    """
    score = 0.0
    for r_pos in receptor_positions:
        for l_pos in ligand_positions:
            distance = np.linalg.norm(r_pos - l_pos)
            if distance > 0:
                score =score + distance # Distance-weighted interaction
    return score
	
# Function to calculate hydrogen bond score from features
def calculate_hbond_score_f3(receptor_features, ligand_features):
    score = 0
    for rec_hbond, lig_hbond in zip(receptor_features['hbond'], ligand_features['hbond']):
        score += rec_hbond * lig_hbond  # Simple multiplication of features
    return score



# Function to calculate hydrophobicity score from features
def calculate_hydrophobicity_score_f3(receptor_features, ligand_features):
    score = 0
    for rec_hphob, lig_hphob in zip(receptor_features['hphob'], ligand_features['hphob']):
        score += rec_hphob * lig_hphob  # Simple multiplication of features
    return score


# Main function to load mesh, extract seed points, align patches, generate poses, and save the best poses
import numpy as np
from Bio.PDB import PDBParser

def transform_and_save_pdb_f3(receptor_structure, ligand_structure, best_poses, receptor_seed_point,output_filename):
    """
    Apply the best transformation poses to the ligand and save both receptor and ligand to a PDB file.

    Parameters:
        receptor_structure: The receptor Biopython Structure object.
        ligand_structure: The original ligand Biopython Structure object.
        best_poses: A list of tuples (rotation_matrix, translation_vector).
        output_filename: The output PDB file path.

    Returns:
        transformed_poses: A list of transformed ligand coordinates (numpy arrays).
    """
    # Extract ligand coordinates from the original structure
    ligand_coords = []
    for atom in ligand_structure.get_atoms():
        ligand_coords.append(atom.get_coord())  # Append each atom's coordinates as a numpy array

    ligand_coords = np.array(ligand_coords)  # Convert list of coordinates to a numpy array for transformation

    if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3:
        raise ValueError(f"ligand_coords should be of shape (N, 3), got shape {ligand_coords.shape}")

    transformed_poses = []

    # Apply each pose to the ligand coordinates
    for rotation_matrix, translation_vector in best_poses:
        # Ensure that rotation_matrix is (3, 3) and translation_vector is (3,)
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"rotation_matrix should be of shape (3, 3), got shape {rotation_matrix.shape}")
        if translation_vector.shape != (3,):
            raise ValueError(f"translation_vector should be of shape (3,), got shape {translation_vector.shape}")
        print("DEBUG PDB trans SAVE ligand_coords before:", ligand_coords)
        print("DEBUG PDB trans SAVE translation_vector:", translation_vector)
        print("DEBUG PDB trans SAVE translation_vector length:", len(translation_vector))		
        # Transform ligand coordinates huyue TTT
        ligand_coords=[item + translation_vector.T - receptor_seed_point.T for item in ligand_coords]
        print("DEBUG PDB trans SAVE ligand_coords after:", ligand_coords) 		
        transformed_coords = np.dot(ligand_coords, rotation_matrix)
		#huyue 20241029-13:23
        print("DEBUG PDB SAVE transformed_coords 0,0,0 after:", transformed_coords)
        transformed_coords=transformed_coords + receptor_seed_point		
        print("DEBUG PDB SAVE transformed_coords final after:", transformed_coords)		
        # Ensure the transformed coordinates are in the correct shape
        if transformed_coords.ndim == 2 and transformed_coords.shape[1] == 3:
            transformed_poses.append(transformed_coords)
        else:
            raise ValueError(f"transformed_coords should be of shape (N, 3), got shape {transformed_coords.shape}")

    # Save the receptor and transformed ligand poses to a PDB file
    save_output_pdb(receptor_structure, ligand_structure, transformed_poses, output_filename)

    return transformed_poses




def transform_and_save_pdb_f1(receptor_structure, ligand_structure, best_poses, output_filename):
    """
    Apply the best transformation poses to the ligand and save both receptor and ligand to a PDB file.

    Parameters:
        receptor_structure: The receptor Biopython Structure object.
        ligand_structure: The original ligand Biopython Structure object.
        best_poses: A list of tuples (rotation_matrix, translation_vector).
        output_filename: The output PDB file path.

    Returns:
        transformed_poses: A list of transformed ligand coordinates (numpy arrays).
    """
    # Extract ligand coordinates from the original structure
    ligand_coords = []
    for atom in ligand_structure.get_atoms():
        ligand_coords.append(atom.get_coord())  # Append each atom's coordinates as a numpy array

    ligand_coords = np.array(ligand_coords)  # Convert list of coordinates to a numpy array for transformation

    # Container for transformed poses
    transformed_poses = []

    # Apply each pose transformation to the ligand coordinates
    for rotation_matrix, translation_vector in best_poses:
        # Ensure the rotation matrix and translation vector are numpy arrays
        if isinstance(rotation_matrix, list):
            rotation_matrix = np.array(rotation_matrix)
        if isinstance(translation_vector, list):
            translation_vector = np.array(translation_vector)

        # Apply the rotation and translation
        transformed_coords = np.dot(ligand_coords, rotation_matrix.T) + translation_vector

        # Add the transformed coordinates to the list of poses
        transformed_poses.append(transformed_coords)

    # Save the receptor and transformed ligand poses to the PDB file
    save_output_pdb(receptor_structure, ligand_structure, transformed_poses, output_filename)

    # Return the transformed ligand poses for further analysis if needed
    return transformed_poses

	


def transform_and_save_pdb(receptor_structure, ligand_structure, best_poses, receptor_seed_points, output_filename):
    """
    Apply the best transformation poses to the ligand and save both receptor and ligand to a PDB file.

    Parameters:
        receptor_structure: The receptor Biopython Structure object.
        ligand_structure: The original ligand Biopython Structure object.
        best_poses: A list of tuples (rotation_matrix, translation_vector).
        receptor_seed_points: A list of receptor seed points corresponding to each pose.
        output_filename: The output PDB file path.

    Returns:
        transformed_poses: A list of transformed ligand coordinates (numpy arrays).
    """
    # Extract ligand coordinates from the original structure
    ligand_coords = []
    for atom in ligand_structure.get_atoms():
        ligand_coords.append(atom.get_coord())

    ligand_coords = np.array(ligand_coords)

    if ligand_coords.ndim != 2 or ligand_coords.shape[1] != 3:
        raise ValueError(f"ligand_coords should be of shape (N, 3), got shape {ligand_coords.shape}")

    transformed_poses = []

    # Check that best_poses and receptor_seed_points have the same length
    if len(best_poses) != len(receptor_seed_points):
        raise ValueError("best_poses and receptor_seed_points must have the same length.")

    # Apply each pose to the ligand coordinates
    for (rotation_matrix, translation_vector), receptor_seed_point in zip(best_poses, receptor_seed_points):
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"rotation_matrix should be of shape (3, 3), got shape {rotation_matrix.shape}")
        if translation_vector.shape != (3,):
            raise ValueError(f"translation_vector should be of shape (3,), got shape {translation_vector.shape}")
        
        #print("DEBUG PDB trans SAVE ligand_coords before:", ligand_coords)
        #print("DEBUG PDB trans SAVE translation_vector:", translation_vector)
        #print("DEBUG PDB trans SAVE translation_vector length:", len(translation_vector))

        # Transform ligand coordinates: apply translation and adjust to receptor seed point
        ligand_translated = [item + translation_vector.T - receptor_seed_point.T for item in ligand_coords]
        #ligand_translated = [item + translation_vector.T for item in ligand_coords]		
        #print("DEBUG PDB trans SAVE ligand_coords after translation to 0,0,0:", ligand_translated)

        transformed_coords = np.dot(ligand_translated, rotation_matrix.T) 
        
        # Translate back to receptor seed point
        transformed_coords = transformed_coords + receptor_seed_point.T
        #print("DEBUG PDB SAVE transformed_coords final:", transformed_coords)

        # Ensure the transformed coordinates are in the correct shape
        if transformed_coords.ndim == 2 and transformed_coords.shape[1] == 3:
            transformed_poses.append(transformed_coords)
        else:
            raise ValueError(f"transformed_coords should be of shape (N, 3), got shape {transformed_coords.shape}")

    # Save the receptor and transformed ligand poses to a PDB file
    save_output_pdb(receptor_structure, ligand_structure, transformed_poses, output_filename)

    return transformed_poses

def save_output_pdb(receptor_structure, ligand_structure, transformed_poses, output_filename):
    """
    Save the receptor structure and each transformed ligand pose to separate PDB files.

    Parameters:
        receptor_structure: The receptor Biopython Structure object.
        ligand_structure: The original ligand Biopython Structure object.
        transformed_poses: A list of transformed ligand coordinates (numpy arrays).
        output_filename: The base output PDB file path. Each file will have an index suffix.
    """
    for pose_idx, pose in enumerate(transformed_poses):
        # Generate a unique filename for each pose by adding an index to the base filename
        output_file = f"{output_filename}_{pose_idx + 1}.pdb"
        
        with open(output_file, 'w') as pdb_file:
            atom_index = 1

            # Save receptor atoms to the PDB file
            for atom in receptor_structure.get_atoms():
                coord = atom.get_coord()  # Extract coordinates of the atom
                resname = str(atom.parent.resname)
                element = str(atom.element)
                pdb_file.write(
                    f"ATOM  {atom_index:5d}  {atom.name:<4}{resname:<3} A{atom.parent.id[1]:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
                )
                atom_index += 1

            # Save the transformed ligand pose to the PDB file
            for atom_idx, atom in enumerate(ligand_structure.get_atoms()):
                # Extract transformed coordinates from the current pose
                atom_coord = pose[atom_idx]

                # Ensure the transformed coordinates are in the correct format
                if isinstance(atom_coord, np.ndarray):
                    atom_coord = atom_coord.tolist()
                elif not isinstance(atom_coord, list):
                    raise TypeError(f"Unexpected type for atom_coord: {type(atom_coord)}")

                # Check that atom_coord has exactly 3 elements and convert to floats
                if len(atom_coord) != 3:
                    raise ValueError(f"atom_coord does not have exactly 3 elements: {atom_coord}")

                try:
                    atom_coord = [float(c) for c in atom_coord]
                except (ValueError, TypeError) as e:
                    raise TypeError(f"Invalid value in atom_coord, unable to convert to float: {atom_coord}") from e

                # Write transformed ligand atom information
                resname = str(atom.parent.resname)
                element = str(atom.element)
                pdb_file.write(
                    f"ATOM  {atom_index:5d}  {atom.name:<4}{resname:<3} B{atom.parent.id[1]:4d}    "
                    f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
                )
                atom_index += 1

        print(f"Saved transformed pose {pose_idx + 1} to {output_file}")
		
	

def save_output_pdb_f3(receptor_structure, ligand_structure, transformed_poses, output_filename):
    """
    Save the receptor structure and transformed ligand poses to a PDB file.

    Parameters:
        receptor_structure: The receptor Biopython Structure object.
        ligand_structure: The original ligand Biopython Structure object.
        transformed_poses: A list of transformed ligand coordinates (numpy arrays).
        output_filename: The output PDB file path.
    """
    with open(output_filename, 'w') as pdb_file:
        atom_index = 1

        # Save receptor atoms to the PDB file
        for atom in receptor_structure.get_atoms():
            coord = atom.get_coord()  # Extract coordinates of the atom
            resname = str(atom.parent.resname)
            element = str(atom.element)
            pdb_file.write(
                f"ATOM  {atom_index:5d}  {atom.name:<4}{resname:<3} A{atom.parent.id[1]:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
            )
            atom_index += 1
        print("DEBUG PDB final SAVE transformed_poses:", transformed_poses)
        print("DEBUG PDB final SAVE length transformed_poses:", len(transformed_poses)) 		
        # Save the transformed ligand poses to the PDB file
        for pose_idx, pose in enumerate(transformed_poses):
            pose_idx = int(pose_idx)  # Ensure pose_idx is an integer
            print("DEBUG PDB final SAVE pose_idx:", pose_idx)
            print("DEBUG PDB final SAVE pose:", pose) 		
            for atom_idx, atom in enumerate(ligand_structure.get_atoms()):
                # Extract transformed coordinates from the current pose
                atom_idx = int(atom_idx)
                atom_coord = pose[atom_idx]
                print("atom coord:", atom_coord)
                print("DEBUG PDB final SAVE atom_idx:", atom_idx)
                print("DEBUG PDB final SAVE atom:", atom)				
                # Ensure that atom_coord is a list of floats
                if isinstance(atom_coord, np.ndarray):
                    atom_coord = atom_coord.tolist()
                elif not isinstance(atom_coord, list):
                    raise TypeError(f"Unexpected type for atom_coord: {type(atom_coord)}")

                # Ensure atom_coord is a length-3 list and convert elements to floats
                if len(atom_coord) != 3:
                    raise ValueError(f"atom_coord does not have exactly 3 elements: {atom_coord}")

                try:
                    atom_coord = [float(c) for c in atom_coord]
                except (ValueError, TypeError) as e:
                    raise TypeError(f"Invalid value in atom_coord, unable to convert to float: {atom_coord}") from e

                # Debug print to verify the values before writing
                #print(f"Debug: atom_coord = {atom_coord}, type = {type(atom_coord)}")

                # Ensure resname and element are strings
                resname = str(atom.parent.resname)
                element = str(atom.element)

                # Write transformed ligand atom information
                pdb_file.write(
                    f"ATOM  {atom_index:5d}  {atom.name:<4}{resname:<3} B{atom.parent.id[1]:4d}    "
                    f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
                )
                atom_index += 1
    

				
def save_output_pdb_f1(receptor_structure, ligand_structure, transformed_poses, output_filename):
    """
    Save the receptor structure and transformed ligand poses to a PDB file.

    Parameters:
        receptor_structure: The receptor Biopython Structure object.
        ligand_structure: The original ligand Biopython Structure object.
        transformed_poses: A list of transformed ligand coordinates (numpy arrays).
        output_filename: The output PDB file path.
    """
    with open(output_filename, 'w') as pdb_file:
        atom_index = 1

        # Save receptor atoms to the PDB file
        for atom in receptor_structure.get_atoms():
            coord = atom.get_coord()  # Extract coordinates of the atom

            # Ensure resname and element are strings
            resname = str(atom.parent.resname)
            element = str(atom.element)

            # Write receptor atom information
            pdb_file.write(
                f"ATOM  {atom_index:5d}  {atom.name:<4} {resname:<3} A{atom.parent.id[1]:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
            )
            atom_index += 1

        # Save the transformed ligand poses to the PDB file
        for pose_idx, pose in enumerate(transformed_poses):
            pose_idx = int(pose_idx)  # Ensure pose_idx is an integer
            
            for atom_idx, atom in enumerate(ligand_structure.get_atoms()):
                # Extract transformed coordinates from the current pose
                
                atom_idx=int(atom_idx)
                atom_coord = pose[atom_idx]
                print("atom coord:", atom_coord)
                # Convert atom coordinates to list if needed
                if isinstance(atom_coord, np.ndarray):
                    atom_coord = atom_coord.tolist()

                # Ensure resname and element are strings
                resname = str(atom.parent.resname)
                element = str(atom.element)

                # Write transformed ligand atom information
                #pdb_file.write(
                 #   f"ATOM  {atom_index:5d}  {atom.name:<4} {resname:<3} B{pose_idx + 1:4d}    "
                 #   f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
                #)
				
                #atom_index += 1
                pdb_file.write(
                    f"ATOM  {atom_index:5d}  {atom.name:<4} {resname:<3} B{pose_idx + 1:4d}    "
                    f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
                )
                atom_index += 1
				

				
def save_output_pdb_f2(receptor_structure, ligand_structure, transformed_poses, output_filename):
    """
    Save the receptor structure and transformed ligand poses to a PDB file.

    Parameters:
        receptor_structure: The receptor Biopython Structure object.
        ligand_structure: The original ligand Biopython Structure object.
        transformed_poses: A list of transformed ligand coordinates (numpy arrays).
        output_filename: The output PDB file path.
    """
    with open(output_filename, 'w') as pdb_file:
        atom_index = 1

        # Save receptor atoms to the PDB file
        for atom in receptor_structure.get_atoms():
            coord = atom.get_coord()  # Extract coordinates of the atom

            # Debugging statements to identify data types
            resname = atom.parent.resname
            element = atom.element
            name = atom.name
            parent_id = atom.parent.id[1]

            # Explicit conversion and additional checks
            if isinstance(resname, (list, tuple)):
                print(f"DEBUG: Found `resname` as list or tuple, converting to string: {resname}")
                resname = str(resname)
            if isinstance(element, (list, tuple)):
                print(f"DEBUG: Found `element` as list or tuple, converting to string: {element}")
                element = str(element)
            if isinstance(name, (list, tuple)):
                print(f"DEBUG: Found `atom.name` as list or tuple, converting to string: {name}")
                name = str(name)
            if isinstance(parent_id, (list, tuple)):
                print(f"DEBUG: Found `parent_id` as list or tuple, converting to integer: {parent_id}")
                parent_id = int(parent_id)

            # Write receptor atom information
            pdb_file.write(
                f"ATOM  {atom_index:5d}  {name:<4} {resname:<3} A{parent_id:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
            )
            atom_index += 1

        # Save the transformed ligand poses to the PDB file
        for pose_idx, pose in enumerate(transformed_poses):
            pose_idx = int(pose_idx)  # Ensure pose_idx is an integer
            
            for atom_idx, atom in enumerate(ligand_structure.get_atoms()):
                # Extract transformed coordinates from the current pose
                atom_coord = pose[atom_idx]

                # Convert atom coordinates to list if needed
                if isinstance(atom_coord, np.ndarray):
                    atom_coord = atom_coord.tolist()

                # Debugging statements to identify data types
                resname = atom.parent.resname
                element = atom.element
                name = atom.name

                # Explicit conversion and additional checks
                if isinstance(resname, (list, tuple)):
                    print(f"DEBUG: Found `resname` as list or tuple, converting to string: {resname}")
                    resname = str(resname)
                if isinstance(element, (list, tuple)):
                    print(f"DEBUG: Found `element` as list or tuple, converting to string: {element}")
                    element = str(element)
                if isinstance(name, (list, tuple)):
                    print(f"DEBUG: Found `atom.name` as list or tuple, converting to string: {name}")
                    name = str(name)

                # Write transformed ligand atom information
                pdb_file.write(
                    f"ATOM  {atom_index:5d}  {name:<4} {resname:<3} B{pose_idx + 1:4d}    "
                    f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}  1.00  0.00           {element:>2}\n"
                )
                atom_index += 1
				
def translate_points(points, translation_vector):
    return points + translation_vector

def save_to_ply_f6(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")				

def save_to_ply_f7(receptor_vertices, seed_points, filename):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(receptor_vertices) + len(seed_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float charge\n")
        f.write("property float hbond\n")
        f.write("property float hphob\n")
        f.write("property float iface\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")

        # Process receptor vertices
        for point in receptor_vertices:
            output_data = [point[0], point[1], point[2], 1, 1, 1, 0, 1, 1, 1]  # iface = 0
            f.write(" ".join(map(str, output_data)) + "\n")

        # Process seed points
        for point in seed_points:
            output_data = [point[0], point[1], point[2], 0, 0, 0, 1, 0, 0, 0]  # iface = 1
            f.write(" ".join(map(str, output_data)) + "\n")

def calculate_faces(num_vertices):
    # Simple example: Create triangular faces for a grid-like structure
    faces = []
    for i in range(num_vertices - 1):  # Adjust as needed
        if (i + 1) % 10 != 0:  # Assuming 10 vertices per row
            faces.append([i, i + 1, i + 10])     # First triangle
            faces.append([i + 1, i + 11, i + 10])  # Second triangle
    return faces

def save_to_ply_f9(receptor_vertices, seed_points, filename):
    all_vertices = np.vstack((receptor_vertices, seed_points))  # Combine vertices
    total_vertices = len(all_vertices)
    
    # Calculate faces based on the total number of vertices
    faces = calculate_faces(total_vertices)

    with open(filename, 'w') as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float charge\n")
        f.write("property float hbond\n")
        f.write("property float hphob\n")
        f.write("property float iface\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")

        # Write vertices
        for point in receptor_vertices:
            output_data = [point[0], point[1], point[2], 1, 1, 1, 0, 1, 1, 1]  # iface = 0
            f.write(" ".join(map(str, output_data)) + "\n")

        for point in seed_points:
            output_data = [point[0], point[1], point[2], 0, 0, 0, 1, 0, 0, 0]  # iface = 1
            f.write(" ".join(map(str, output_data)) + "\n")

        # Write face data
        for face in faces:
            face_line = f"{len(face)} " + " ".join(map(str, face))
            f.write(face_line + "\n")

def save_to_ply(receptor_vertices, seed_points, filename):
    # Convert seed_points to a set for faster lookup
    seed_set = set(map(tuple, seed_points))

    with open(filename, 'w') as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(receptor_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float charge\n")
        f.write("property float hbond\n")
        f.write("property float hphob\n")
        f.write("property float iface\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")

        # Write receptor vertices with iface based on presence in seed_points
        for point in receptor_vertices:
            iface = 1 if tuple(point) in seed_set else 0
            output_data = [point[0], point[1], point[2], -1.22518, -0.902427, -1.6, iface, 0.979109, 0.0543578, 0.195937]
            f.write(" ".join(map(str, output_data)) + "\n")				
				
def main():
    # Step 1: Load mesh files and find seed points for receptor and ligand
    receptor_mesh_path = '1AVW_A.ply'
    ligand_mesh_path = '1AVW_B.ply'
    receptor_pdb_path = '1AVW_A.pdb'
    ligand_pdb_path = '1AVW_B.pdb'

    # Check if input files exist
    if not os.path.exists(receptor_mesh_path):
        raise FileNotFoundError(f"Receptor mesh file '{receptor_mesh_path}' not found.")
    if not os.path.exists(ligand_mesh_path):
        raise FileNotFoundError(f"Ligand mesh file '{ligand_mesh_path}' not found.")
    if not os.path.exists(receptor_pdb_path):
        raise FileNotFoundError(f"Receptor PDB file '{receptor_pdb_path}' not found.")
    if not os.path.exists(ligand_pdb_path):
        raise FileNotFoundError(f"Ligand PDB file '{ligand_pdb_path}' not found.")

    # Load meshes
    receptor_mesh, receptor_features = load_ply_features(receptor_mesh_path)
    ligand_mesh, ligand_features = load_ply_features(ligand_mesh_path)

    # Extract vertices from the loaded mesh data
    #receptor_vertices = np.array(receptor_mesh.vertices)
    #ligand_vertices = np.array(ligand_mesh.vertices)
    receptor_vertices = np.array(receptor_mesh)
    ligand_vertices = np.array(ligand_mesh)
    # Step 2: Find seed points for receptor and ligand and set concave-convex
    receptor_seed_points, receptor_sorted_abs_differences, receptor_sorted_eigenvalues, receptor_sorted_eigenvectors = find_seed_points_energy(receptor_vertices, receptor_features,con=0)
    ligand_seed_points, ligand_sorted_abs_differences, ligand_sorted_eigenvalues, ligand_sorted_eigenvectors = find_seed_points_energy(ligand_vertices, ligand_features,con=1)
    #print("Receptor seed points:", receptor_seed_points)
    #print("Ligand seed points:", ligand_seed_points)




    # Initialize the parser
    parser = PDBParser()
    # Step 5: Obtain binding poses - for now we use only one pair of seed points
    receptor_structure = parser.get_structure("receptor", receptor_pdb_path)
    ligand_structure = parser.get_structure("ligand", ligand_pdb_path)
    #test_seed_pair = (receptor_seed_points[110], ligand_seed_points[110])
    #binding_poses, rotation_matrix, translation_vector, receptor_seed_point = obtain_binding_poses(receptor_vertices, ligand_vertices, test_seed_pair[0], test_seed_pair[1], receptor_descriptors[110], ligand_descriptors[110], receptor_extremes[110], ligand_extremes[110])
    #print("Binding poses:", binding_poses)
    all_binding_poses = []
    all_rotation_matrices = []
    all_translation_vectors = []
    all_receptor_seed_points = []
    all_ligand_seed_points = []
     # 假设 seed 点的数量是 N
    N = len(receptor_seed_points)  # 或 len(ligand_seed_points)，二者长度应相同
    M = len(ligand_seed_points)
    print("receptor_seed_points length",N)
    print("ligand_seed_points length",M)	
    #for i in range(N):

    for i in range(50):	
    # 获取当前的 seed pair
        #for j in range(M):
        for j in range(50):		

    # 调用 obtain_binding_poses 计算绑定姿态
            binding_poses, rotation_matrices, translation_vector, receptor_seed_point, ligand_seed_point = obtain_binding_poses_hessian(
            receptor_vertices,
            ligand_vertices,
            receptor_seed_points[i],
            ligand_seed_points[j],
            receptor_sorted_eigenvectors[i],
            ligand_sorted_eigenvectors[j]
    )
        for rotation_matrix in rotation_matrices:
            all_binding_poses.append(binding_poses)
            all_rotation_matrices.append(rotation_matrix)
            all_translation_vectors.append(translation_vector)
            all_receptor_seed_points.append(receptor_seed_point)
            all_ligand_seed_points.append(ligand_seed_point)
			
		
			
    # 将结果添加到列表中

    #print(f"Binding poses for seed {i}:", binding_poses)

    # 最终可以得到所有 seed 的绑定姿态结果在 all_binding_poses 列表中

    #best_poses=[]
    #best_seed_points=[]
    #pose=(rotation_matrix,translation_vector)
    #best_poses.append(pose)
    #best_seed_points.append(receptor_seed_point)	
    best_poses, best_scores , best_seed_points = score_poses_with_seed_interaction(receptor_mesh, ligand_mesh, all_rotation_matrices, all_translation_vectors, receptor_features, ligand_features, all_receptor_seed_points, all_ligand_seed_points)
    #print("Scored poses:", best_scores)
	
    transform_and_save_pdb(receptor_structure, ligand_structure, best_poses, best_seed_points, 'output.pdb')
    print("Output PDB saved to output.pdb")
    # Score the poses	
	
if __name__ == "__main__":
    main()
		
		
