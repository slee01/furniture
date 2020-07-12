import torch
import torch.utils.data


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, env_id, file_name):
        all_trajectories = torch.load(file_name)

        if env_id.startswith("MountainToy"):
            num_trajectories, subsample_frequency = 256, 4
        elif env_id.startswith("Fetch"):
            num_trajectories, subsample_frequency = 512, 4
        else:
            num_trajectories, subsample_frequency = 32, 3

        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        start_idx = torch.randint(0, all_trajectories['states'].size(1) % subsample_frequency,
                                  size=(num_trajectories,)).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}

        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []

        for j in range(self.length):

            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories['actions'][traj_idx][i]
