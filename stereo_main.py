import numpy as np
import cv2
from cv2 import cuda
from matplotlib import pyplot as plt


# cuda.printCudaDeviceInfo(0)


class StereoWrapper:
    def __init__(self, num_disparities: int = 128, block_size: int = 25, bp_ndisp: int = 64, min_disparity: int = 16, uniqueness_ratio: int = 5) -> None:
        self.stereo_bm_cuda = cuda.createStereoBM(numDisparities=num_disparities, blockSize=block_size)
        self.stereo_bp_cuda = cuda.createStereoBeliefPropagation(ndisp=bp_ndisp)
        self.stereo_bcp_cuda = cuda.createStereoConstantSpaceBP(min_disparity)
        self.stereo_sgm_cuda = cuda.createStereoSGM(minDisparity=min_disparity, numDisparities=num_disparities, uniquenessRatio=uniqueness_ratio)

    @staticmethod
    def __numpy_to_gpumat(np_image: np.ndarray) -> cv2.cuda_GpuMat:
        image_cuda = cv2.cuda_GpuMat()
        image_cuda.upload(cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY))
        return image_cuda

    def compute_disparity(self, left_img: np.ndarray, right_img: np.ndarray, algorithm_name: str = "stereo_sgm_cuda") -> np.ndarray:
        algorithm = getattr(self, algorithm_name)
        left_cuda = self.__numpy_to_gpumat(left_img)
        right_cuda = self.__numpy_to_gpumat(right_img)

        if algorithm_name == "stereo_sgm_cuda":
            disparity_sgm_cuda_2 = cv2.cuda_GpuMat()
            disparity_sgm_cuda_1 = algorithm.compute(left_cuda, right_cuda, disparity_sgm_cuda_2)
            return disparity_sgm_cuda_1.download()

        else:
            disparity_cuda = algorithm.compute(left_cuda, right_cuda, cv2.cuda_Stream.Null())
            return disparity_cuda.download()


left_img = cv2.imread("G:\\Workspace\\backup_20200915\\data_scene_flow_multiview\\testing\\image_2\\L000005_15.png")
right_img = cv2.imread("G:\\Workspace\\backup_20200915\\data_scene_flow_multiview\\testing\\image_3\\R000005_15.png")
wrapper = StereoWrapper()
disparity_map = wrapper.compute_disparity(left_img, right_img)
plt.imshow(disparity_map, 'gist_rainbow')
plt.show()
