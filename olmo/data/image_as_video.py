import dataclasses
from typing import Callable, Tuple, Optional

import numpy as np
from scipy.stats import beta, dirichlet, norm, multinomial

from olmo.config import BaseConfig, D
from olmo.data.image_preprocessor import siglip_resize_and_pad, select_tiling


def sample_stick_breaking(alpha, K):
    betas = beta.rvs(1, alpha, size=K)

    # Calculate stick-breaking weights using vectorized operations
    # First, transform betas into (1-beta) values
    one_minus_betas = 1 - betas

    # Compute the cumulative product to get the remaining stick portions
    # We need to exclude the last beta value and prepend 1
    cumprod_shifted = np.cumprod(one_minus_betas[:-1])
    remaining_sticks = np.concatenate(([1.0], cumprod_shifted))

    # Multiply by betas to get the weights
    weights = remaining_sticks * betas

    # Normalize weights to ensure they sum to 1
    weights /= np.sum(weights)

    return weights


@dataclasses.dataclass
class OutsideInPath:
    on: int
    xs: int
    xe: int
    ys: int
    ye: int
    top: bool
    left: bool
    indices: np.ndarray

    @staticmethod
    def init(w, h, rng):
        return OutsideInPath(
            0, 0, w, 0, h,
            rng.random()<0.5,
            rng.random()<0.5,
            np.zeros((w, h), dtype=np.int32),
        )

    @staticmethod
    def build_path(w, h, rng, p_spiral=0, p_zigzg=0) -> np.ndarray:
        path = OutsideInPath.init(w, h, rng)
        if p_zigzg or p_spiral:
            mode_r = rng.random()
            if mode_r < p_zigzg:
                path.zigzag(rng)
            elif mode_r < (p_zigzg + p_spiral):
                path.spiral(rng)
            else:
                path.mix(rng)
        else:
            path.mix(rng)
        indices = path.get_indices()
        if rng.random() < 0.5:
            indices = indices[::-1]
        return indices

    def get_indices(self):
        indices = np.indices(self.indices.shape).reshape([2, -1]).T
        ordered = np.zeros_like(indices)
        ordered[self.indices.ravel()] = indices
        return ordered

    def step_v(self):
        ix = np.arange(self.on, self.on+self.xe-self.xs)
        self.on += len(ix)
        if not self.top:
            ix = ix[::-1]
        if self.left:
            self.indices[self.xs:self.xe, self.ys] = ix
            self.ys += 1
        else:
            self.indices[self.xs:self.xe, self.ye-1] = ix
            self.ye -= 1
        self.top = not self.top

    def step_h(self):
        ix = np.arange(self.on, self.on+self.ye-self.ys)
        self.on += len(ix)
        if not self.left:
            ix = ix[::-1]
        if self.top:
            self.indices[self.xs, self.ys:self.ye] = ix
            self.xs += 1
        else:
            self.indices[self.xe-1, self.ys:self.ye] = ix
            self.xe -= 1
        self.left = not self.left

    def step(self, h):
        if h:
            self.step_h()
            return self.xs == self.xe
        else:
            self.step_v()
            return self.ys == self.ye

    def spiral(self, rng):
        step_h = rng.random() < 0.5
        while not self.step(step_h):
            step_h = not step_h

    def zigzag(self, rng):
        step_h = rng.random() < 0.5
        while not self.step(step_h):
            pass

    def mix(self, rng):
        while not self.step(rng.random() < 0.5):
            pass


if __name__ == '__main__':
    rng = np.random
    path = OutsideInPath.init(8, 5, rng)
    path.mix(rng)
    print(path.indices)
    print(path.get_indices(rng))


@dataclasses.dataclass
class ImageAsVideoConfig(BaseConfig):
    min_repeats: int
    min_overlap: int
    min_step_size: Optional[int] = None
    max_pixels: Optional[int] = None
    max_crops: Optional[int] = None
    preferred_step_size: Optional[int] = None
    max_repeats: Optional[int] = None
    path_mode: str = "zigzag"
    frame_mode: str = "steps"

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "build_source_ids" in config:
            del config["build_source_ids"]
        return config

    def build(self, max_frames, vit_config):
        if vit_config.resize_mode == "siglip":
            resize_fn = siglip_resize_and_pad
        else:
            raise NotImplementedError(vit_config.resize_mode)
        return ImagePan(
            preferred_step_size=self.preferred_step_size,
            patch_size=vit_config.image_patch_size,
            crop_size=vit_config.image_default_input_size,
            min_repeats=self.min_repeats,
            min_overlap=self.min_overlap,
            max_pixels=self.max_pixels,
            max_frames=max_frames,
            min_step_size=self.min_step_size,
            resize_fn=resize_fn,
            max_crops=self.max_crops,
            max_repeats=self.max_repeats,
            path_mode=self.path_mode,
            frame_mode=self.frame_mode
        )


@dataclasses.dataclass
class ImagePan:
    patch_size: int
    max_frames: int
    min_repeats: int
    min_overlap: int
    crop_size: Tuple[int, int]
    resize_fn: Callable
    min_step_size: Optional[int] = None
    preferred_step_size: int = 0
    max_repeats: Optional[int] = None
    max_pixels: int = None
    max_crops: Optional[int] = None
    path_mode: str = "zigzag"
    frame_mode: str = "steps"

    def build_path(self, h, w, rng: np.random.RandomState):
        if self.path_mode == "outside-in-v1":
            return OutsideInPath.build_path(h, w, rng, 0.05, 0.1)
        elif self.path_mode == "left-to-right-up-down":
            path = []
            for y in range(h):
                for x in range(w):
                    path.append([y, x])
            return np.array(path)
        elif self.path_mode == "zigzag":
            path = []
            for x in range(h):
                for y in range(w):
                    if x % 2 == 1:
                        y = w - y - 1
                    path.append([x, y])
            return np.array(path)
        else:
            raise NotImplementedError(self.path_mode)

    def get_candidate_step_sizes(self, h_r, c_h):
        if h_r == 0:
            return np.array([[0, 0]])
        else:
            h_n_steps = np.unique(h_r // np.arange(self.min_step_size if self.min_step_size else 1, h_r + 1))
            h_n_steps = h_n_steps[h_n_steps <= self.max_frames-1-self.min_repeats]  # -1 for the initial frames
            h_step_size = h_r // h_n_steps + (h_r % h_n_steps != 0)
            keep = h_step_size <= (c_h//self.patch_size - self.min_overlap)
            return np.stack([h_step_size[keep], h_n_steps[keep]+1], -1)

    def build_steps(self, n_steps, value, max_step):
        if n_steps == 0:
            return np.array([0])
        n_steps -= 1  # for the first frame
        h_steps = np.full([n_steps], value // n_steps)
        r = value % n_steps
        if r:
            h_steps[:value % n_steps] += 1
            np.random.shuffle(h_steps)
        assert h_steps.max() <= max_step
        h_steps = np.cumsum(h_steps)
        assert h_steps[-1] == value
        return np.pad(h_steps, [1, 0])

    def __call__(self, image, rng):
        c_h, c_w = self.crop_size
        d = self.patch_size
        h, w = image.shape[:2]

        if self.max_crops:
            # Resize the image to the same size it would be if we were using max-crop tiling
            total_margin_pixels = d * 8
            crop_patches = c_h // d
            crop_window_patches = crop_patches - 8
            crop_window_size = crop_window_patches * d
            tiling = select_tiling(
                h - total_margin_pixels,
                w - total_margin_pixels,
                crop_window_size,
                self.max_crops
            )
            target_h, target_w = [
                (tiling[0]*crop_window_size+total_margin_pixels),
                (tiling[1]*crop_window_size+total_margin_pixels)
            ]
        else:
            if self.max_pixels and (h * w) > self.max_pixels:
                ratio = np.sqrt(self.max_pixels / (h * w))
                h = int(h*ratio)
                w = int(w*ratio)
            target_h = max(h, c_h)
            target_h = (target_h + self.patch_size - 1) // d * d
            target_w = max(w, c_w)
            target_w = (target_w + self.patch_size - 1) // d * d

        # FIXME resize extremely large images
        image, _ = self.resize_fn(image, [target_h, target_w])

        if self.frame_mode == "crops":
            h_steps = np.arange(tiling[0]) * crop_window_size
            w_steps = np.arange(tiling[1]) * crop_window_size
            assert h_steps.max() + c_h == target_h
            assert w_steps.max() + c_w == target_w
        elif self.frame_mode == "steps":
            h_r = (target_h - c_h) // d
            h_step_sizes = self.get_candidate_step_sizes(h_r, c_h)
            w_r = (target_w - c_w) // d
            w_step_sizes = self.get_candidate_step_sizes(w_r, c_w)

            candidates = np.stack([
                np.tile(h_step_sizes[:, None, :], [1, len(w_step_sizes), 1]),
                np.tile(w_step_sizes[None, :, :], [len(h_step_sizes), 1, 1]),
            ], -2).reshape([-1, 2, 2])
            candidates = candidates[np.prod(candidates[:, :, 1], -1) <= self.max_frames-self.min_repeats]

            scores = np.square(np.maximum(candidates[:, :, 0], self.preferred_step_size)).sum(-1)
            ix_candidates = np.argwhere(scores == scores.min())[:, 0]
            ix = ix_candidates[rng.randint(0, len(ix_candidates))]
            (_, h_steps), (_, w_steps) = candidates[ix]
            assert h_steps*w_steps <= self.max_frames
            h_steps = self.build_steps(h_steps, target_h - c_h, c_h - self.min_overlap*d)
            w_steps = self.build_steps(w_steps, target_w - c_w, c_w - self.min_overlap*d)
        else:
            raise NotImplementedError(self.frame_mode)

        crops = np.zeros([len(h_steps), len(w_steps), c_h, c_w, 3])
        source_ids = np.zeros([len(h_steps), len(w_steps), c_h//d, c_w//d], dtype=np.int32)
        image_ids = np.arange(target_h*target_w//(d*d)).reshape([target_h//d, target_w//d])
        for x, x0 in enumerate(h_steps):
            for y, y0 in enumerate(w_steps):
                crops[x, y] = image[x0:x0+c_h, y0:y0+c_w]
                source_ids[x, y] = image_ids[x0//d:(x0+c_h)//d, y0//d:(y0+c_w)//d]

        path = self.build_path(len(h_steps), len(w_steps), rng)
        crops = crops[path[:, 0], path[:, 1]].reshape([-1, c_h, c_w, 3])
        source_ids = source_ids[path[:, 0], path[:, 1]].reshape([-1, c_h//d, c_w//d])

        n_frames = len(h_steps)*len(w_steps)
        if n_frames < self.max_frames:
            n_target_frames = rng.randint(n_frames, self.max_frames)
            n_repeat = n_target_frames - n_frames
            if self.max_repeats is not None:
                n_repeat = min(self.max_repeats, n_repeat)
        else:
            n_repeat = 0
        if n_repeat:
            weights = sample_stick_breaking(5, K=crops.shape[0])
            np.random.shuffle(weights)
            repeats = multinomial(n_repeat, weights).rvs()[0]
            crops = np.repeat(crops, repeats+1, axis=0)
            if source_ids is not None:
                source_ids = np.repeat(source_ids, repeats+1, axis=0)
        assert len(crops) <= self.max_frames
        return crops, np.arange(len(crops)), 1, dict(source_ids=source_ids, image_size=[target_h, target_w])


if __name__ == '__main__':
    print()
