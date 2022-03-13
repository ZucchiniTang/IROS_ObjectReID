class IrregularSampler(Sampler):
    def __init__(self, data_source, num_instances, video=False):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.video = video
        if self.video:
            for index, (_, pid, _, _) in enumerate(data_source):
                self.index_dic[pid].append(index)
        else:
            for index, (_, pid, _) in enumerate(data_source):
                self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=len(t), replace=False)
            ret.extend(t)
        return iter(ret)
