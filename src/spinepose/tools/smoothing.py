import numpy as np
from OneEuroFilter import OneEuroFilter


class KeypointSmoothing:
    def __init__(
        self,
        num_keypoints,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ):
        """Initialize the PoseFiltering class.

        Args:
            num_keypoints (int): Number of keypoints to filter.
            freq (float): An estimate of the frequency in Hz of the signal (> 0), if timestamps are not available.
            mincutoff (float, optional): Min cutoff frequency in Hz (> 0). Lower values allow to remove more jitter.
            beta (float, optional): Parameter to reduce latency (> 0).
            dcutoff (float, optional): Used to filter the derivates. 1 Hz by default. Change this parameter if you know what you are doing.

        Raises:
            ValueError: If one of the parameters is not >0
        """
        kwargs = dict(
            freq=freq,
            mincutoff=mincutoff,
            beta=beta,
            dcutoff=dcutoff,
        )
        self.filters = [
            {
                "filter_x": OneEuroFilter(**kwargs),
                "filter_y": OneEuroFilter(**kwargs),
            }
            for _ in range(num_keypoints)
        ]

    def filter_keypoint(self, point, filters):
        """Filter a single keypoint.

        Args:
            point (np.array): The 3D keypoint to filter. Shape: (3,)
            filters (dict): Filters for x and y coordinates.

        Returns:
            filtered_point (np.array): The filtered 3D keypoint. Shape: (3,)
        """
        filtered_x = filters["filter_x"](point[0])
        filtered_y = filters["filter_y"](point[1])
        return np.array([filtered_x, filtered_y])

    def __call__(self, keypoints):
        """Filter all keypoints in a pose.

        Only smooth the first person.

        Args:
            keypoints_3d (np.array): The 3D keypoints to filter. Shape: (K, 2)
            timestamp (float): The timestamp in seconds.

        Returns:
            filtered_keypoints (np.array): The filtered 3D keypoints. Shape: (num_keypoints, 3)
        """
        return np.array(
            [
                self.filter_keypoint(kpt, filters)
                for kpt, filters in zip(keypoints, self.filters)
            ]
        )
