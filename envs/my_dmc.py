import gymnasium as gym
from dm_control import suite
import numpy as np
import cv2


class DMControl(gym.Env):
    """
    Wrapper for DeepMind Control Suite environments with camera following agent.
    Maintains compatibility with the existing Atari-style interface.
    """
    
    def __init__(
        self,
        domain_name,
        task_name,
        action_repeat=2,
        size=(64, 64),
        camera_id=0,
        seed=None,
        length=1000,
    ):
        """
        Args:
            domain_name: DMControl domain (e.g., 'cheetah', 'walker', 'humanoid')
            task_name: Task name (e.g., 'run', 'walk', 'stand')
            action_repeat: Number of times to repeat each action
            size: Image observation size (height, width)
            camera_id: Camera to use for rendering (0 = tracking camera in most envs)
            seed: Random seed
            length: Maximum episode length in steps
        """
        assert size[0] == size[1], "Only square images supported"
        
        self._domain_name = domain_name
        self._task_name = task_name
        self._repeat = action_repeat
        self._size = size
        self._camera_id = camera_id
        self._length = length
        self._seed = seed
        self._random = np.random.RandomState(seed)
        
        # Load DMControl environment
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={'random': seed}
        )
        
        # Get action spec
        self._action_spec = self._env.action_spec()
        self._action_dim = self._action_spec.shape[0]
        
        self._done = True
        self._step_count = 0
        
        # Action space (continuous for DMControl)
        self.action_space = gym.spaces.Box(
            low=self._action_spec.minimum,
            high=self._action_spec.maximum,
            shape=(self._action_dim,),
            dtype=np.float32
        )
        self.action_space.discrete = False  # Mark as continuous
        
        # Observation space
        img_shape = self._size + (3,)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
        })
        
        self.reward_range = [-np.inf, np.inf]
        self.metadata = {}
        
    def _get_obs(self, time_step):
        """Render RGB image from physics."""
        # Render from the specified camera
        img = self._env.physics.render(
            height=self._size[0],
            width=self._size[1],
            camera_id=self._camera_id
        )
        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 255)).astype(np.uint8)
        return img
    
    def reset(self):
        """Reset environment."""
        time_step = self._env.reset()
        self._done = False
        self._step_count = 0
        
        obs = self._get_obs(time_step)
        info = {
            'is_first': True,
            'is_terminal': False,
            'episode_frame_number': 0
        }
        
        return obs, info
    
    def step(self, action):
        """
        Step environment with action repeat.
        
        Args:
            action: Continuous action array of shape (action_dim,)
        """
        # Ensure action is in valid range
        action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)
        
        total_reward = 0.0
        
        # Action repeat loop
        for _ in range(self._repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward or 0.0
            self._step_count += 1
            
            if time_step.last():
                break
        
        # Get observation
        obs = self._get_obs(time_step)
        
        # Determine termination
        my_truncated = self._length and self._step_count >= self._length
        self._done = time_step.last() or my_truncated
        
        # Build info dict
        info = {
            'is_first': False,
            'is_terminal': time_step.last(),  # True episode termination
            'episode_frame_number': self._step_count * self._repeat,
            'discount': time_step.discount
        }
        
        return obs, total_reward, self._done, info
    
    def close(self):
        """Close environment."""
        return self._env.close()


if __name__ == '__main__':
    # Test DMControl environment
    env = DMControl(
        domain_name='cheetah',
        task_name='run',
        action_repeat=2,
        size=(64, 64),
        camera_id=0,
        seed=42,
        length=1000
    )
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action space discrete: {env.action_space.discrete}")
    
    import time
    start_time = time.time()
    
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs, info = env.reset()
    
    end_time = time.time()
    env.close()
    print(f"Time taken for 1,000 iterations: {end_time - start_time:.2f} seconds")