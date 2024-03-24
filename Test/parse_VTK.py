import vtk
import numpy as np

# 1. 读取 vts 文件和 STL 文件
reader_vts = vtk.vtkXMLStructuredGridReader()
reader_vts.SetFileName("IBM.vts")
reader_vts.Update()

reader_stl = vtk.vtkSTLReader()
reader_stl.SetFileName("X59-quesst.stl")
reader_stl.Update()

# 2. 获取 vts 网格和 STL 对象的数据
structured_grid = reader_vts.GetOutput()
stl_polydata = reader_stl.GetOutput()
num_points = structured_grid.GetNumberOfPoints()

# 3. 缩小 STL 对象
transform = vtk.vtkTransform()
transform.Scale(0.1, 0.1, 0.1)

transform_filter = vtk.vtkTransformFilter()
transform_filter.SetInputData(stl_polydata)
transform_filter.SetTransform(transform)
transform_filter.Update()

# 4. 使用 vtkSelectEnclosedPoints 来检测网格中的点是否在 STL 对象内部
select_enclosed_points = vtk.vtkSelectEnclosedPoints()
select_enclosed_points.SetSurfaceData(transform_filter.GetOutput())
select_enclosed_points.SetInputData(structured_grid)
select_enclosed_points.SetTolerance(1e-9)

# 5. 运行选择
select_enclosed_points.Update()

# 6. 创建标记数组
tag_array = np.zeros((num_points), dtype=int)

symmetrical_points = np.zeros((num_points, 3), dtype=np.float64)

dist_to_wall = np.zeros((num_points), dtype=np.float64)

# 使用 vtkCellLocator 来找到 STL 表面上的最近点
locator = vtk.vtkCellLocator()
locator.SetDataSet(transform_filter.GetOutput())
locator.BuildLocator()

# 7. 设置标记数组的值
for i in range(num_points):
    if select_enclosed_points.IsInside(i):
        tag_array[i] = 1
        point = structured_grid.GetPoint(i)
        closest_point = [0, 0, 0]
        cn = vtk.vtkGenericCell()
        cid = vtk.reference(0)
        sid = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        locator.FindClosestPoint(point, closest_point, cn, cid, sid, dist2)
        symmetrical_point = [2 * closest_point[j] - point[j] for j in range(3)]
        symmetrical_points[i, :] = symmetrical_point
        dist_to_wall[i] = dist2

dist_to_wall = np.sqrt(dist_to_wall)

# 写出结果
f = open("data.txt", "w")

for i in range(num_points):
    f.write(str(tag_array[i]) + "\t" +
            str(symmetrical_points[i, 0]) + "\t" + str(symmetrical_points[i, 1]) + "\t" + str(symmetrical_points[i, 2]) + "\t" + 
            str(dist_to_wall[i])+"\n")

f.close()
