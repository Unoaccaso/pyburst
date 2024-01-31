"""
Copyright (C) 2024 Riccardo Felicetti <https://github.com/Unoaccaso>

Created Date: Saturday, January 27th 2024, 6:44:19 pm
Author: Riccardo Felicetti

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE. See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https: //www.gnu.org/licenses/>.
"""
# system libs for package managing
import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)


from utils.commons import BLOCK_SHAPE, CONFIG

# cuda section
from cupyx import jit

# cuda
import cupy
import cupy.typing
import numpy


@jit.rawkernel()
def ray_casting(
    image: cupy.typing.NDArray,
    threshold: numpy.float32,
    fill: bool,
    boundary_only_image: cupy.typing.NDArray,
):
    y_axis = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    x_axis = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y

    if (y_axis < image.shape[0]) and (x_axis < image.shape[1]):
        # is a candidate
        if image[y_axis, x_axis] > threshold:
            # not on the frame
            if (
                (y_axis > 0)
                and (y_axis < (image.shape[0] - 1))
                and (x_axis > 0)
                and (x_axis < (image.shape[1] - 1))
            ):
                # is on the border
                if fill:
                    boundary_only_image[y_axis, x_axis] = image[x_axis, y_axis]
                else:
                    if (
                        (image[y_axis - 1, x_axis] < threshold)
                        or (image[y_axis, x_axis + 1] < threshold)
                        or (image[y_axis + 1, x_axis] < threshold)
                        or (image[y_axis, x_axis - 1] < threshold)
                    ):
                        nor_cuspid_nor_column = (
                            (image[y_axis + 1, x_axis] < threshold)
                            or (image[y_axis - 1, x_axis] < threshold)
                        ) and (
                            (image[y_axis + 1, x_axis] > threshold)
                            or (image[y_axis - 1, x_axis] > threshold)
                        )
                        # is not a cuspid point
                        if nor_cuspid_nor_column:
                            boundary_only_image[y_axis, x_axis] = int(1)

            # exclude corners
            elif (
                ((x_axis == 0) and (y_axis == 0))
                or ((x_axis == 0) and (y_axis == (image.shape[0] - 1)))
                or ((x_axis == (image.shape[1] - 1)) and (y_axis == 0))
                or (
                    (x_axis == (image.shape[1] - 1))
                    and (y_axis == (image.shape[0] - 1))
                )
            ):
                pass

            else:
                # on the x axis
                if (y_axis == 0) and (image[y_axis + 1, x_axis] > threshold):
                    boundary_only_image[y_axis, x_axis] = int(1)
                elif (y_axis == (image.shape[0] - 1)) and (
                    image[y_axis - 1, x_axis] > threshold
                ):
                    boundary_only_image[y_axis, x_axis] = int(1)

                # on the y axis (more complicated)
                elif (x_axis == 0) and (image[y_axis, x_axis + 1] > threshold):
                    nor_cuspid_nor_column = (
                        (image[y_axis + 1, x_axis] < threshold)
                        or (image[y_axis - 1, x_axis] < threshold)
                    ) and (
                        (image[y_axis + 1, x_axis] > threshold)
                        or (image[y_axis - 1, x_axis] > threshold)
                    )
                    if nor_cuspid_nor_column:
                        boundary_only_image[y_axis, x_axis] = int(1)
                elif (x_axis == (image.shape[1] - 1)) and (
                    image[y_axis, x_axis - 1] > threshold
                ):
                    nor_cuspid_nor_column = (
                        (image[y_axis + 1, x_axis] < threshold)
                        or (image[y_axis - 1, x_axis] < threshold)
                    ) and (
                        (image[y_axis + 1, x_axis] > threshold)
                        or (image[y_axis - 1, x_axis] > threshold)
                    )
                    if nor_cuspid_nor_column:
                        boundary_only_image[y_axis, x_axis] = int(1)


def find_boundaries(
    image,
    threshold: numpy.float32 = numpy.float32(
        CONFIG["computation.parameters"]["EnergyThreshold"]
    ),
    fill=False,
):
    contour = cupy.zeros_like(image, dtype=numpy.float32)
    grid_shape = (
        image.shape[0] // BLOCK_SHAPE[0] + 1,  # X
        image.shape[1] // BLOCK_SHAPE[1] + 1,  # Y
    )
    block_shape = BLOCK_SHAPE

    ray_casting[grid_shape, block_shape](image, threshold, fill, contour)

    return contour
