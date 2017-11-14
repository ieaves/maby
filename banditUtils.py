from scipy.stats import beta


class banditStrategy():
    def initializeBandits(self, baseStructure, bandit):
        if not isinstance(baseStructure, dict):
            return banditImplementation(baseStructure, bandit)

        if isinstance(baseStructure, dict):
            keys = list(baseStructure.keys())
            self.keys = keys
            for key in keys:
                subStructure = baseStructure[key]
                if isinstance(subStructure, dict):
                    setattr(self, key, banditStrategy(subStructure, bandit))
                else:
                    setattr(self, key, self.initializeBandits(subStructure, bandit))

    def __init__(self, baseStructure, bandit):
        self.initializeBandits(baseStructure, bandit)
        self.bandit = banditImplementation(self.keys, bandit)

    def get(self, key):
        return getattr(self, key)

    def evaluate(self, path=None):
        def traversalStrategy(obj, path):
            if isinstance(obj, banditStrategy):
                path.append(obj.bandit.evaluate())
                obj = getattr(obj, path[-1])
                path = traversalStrategy(obj, path)
            else:
                path.append(obj.evaluate())

            return path

        if not path:
            key = self.bandit.evaluate()
            path = [key]
            obj = getattr(self, path[-1])

        return traversalStrategy(obj, path)

    def update(self, path, outcome, *args, **kwargs):
        self.bandit.update(path[0], outcome, *args, **kwargs)
        getattr(self, path[0]).update(path[1:], outcome, *args, **kwargs)

    def updateBanditPriors(self, updateObj):
        def dict_updater(obj, update):
            for key in update.keys():
                getattr(self, key).updateBanditPriors(update[key])

        updateObj = updateObj if isinstance(updateObj, list) else [updateObj]
        [dict_updater(self, item) for item in updateObj]

    def performance(self):
        return self.bandit.performance()


class banditImplementation():
    def __init__(self, keys, bandit):
        self.bandit = bandit(keys)
        self.keys = keys

    def evaluate(self):
        return self.keys[self.bandit.select_arm()]

    def update(self, key, result, *args, **kwargs):
        if isinstance(key, list):
            if len(key) > 1:
                for k in key:
                    self.bandit.update(self.keys.index(k))
            else:
                key = key[0]
        self.bandit.update(self.keys.index(key), result, *args, **kwargs)

    def updateBanditPriors(self, updateObj):
        if 'bandit' in updateObj.keys():
            updateObj = updateObj['bandit']

        for key, val in updateObj.items():
                ind = self.keys.index(key)
                for k, v in val.items():
                    getattr(self.bandit, k)[ind] = v

    def performance(self):
        return self.bandit.metric

    def getTopKeys(self, n):
        from numpy import array

        top_indices = array(self.bandit.rewards).argsort()[-n:][::-1]
        return [{self.keys[ind]: self.bandit.trials[ind]} for ind in top_indices]


class BetaBandit(object):
    def __init__(self, keys, *args, **kwargs):
        self.defaultBandit(keys)

    def defaultBandit(self, keys):
        setattr(self, 'metric', [])
        setattr(self, 'num_options', len(keys))
        setattr(self, 'trials', [0] * self.num_options)
        setattr(self, 'rewards', [0] * self.num_options)

        default_prior = [1.0, 1.0]
        setattr(self, 'prior', [default_prior for i in range(self.num_options)])

    def update(self, trial_id, success, *args, **kwargs):
        self.trials[trial_id] = self.trials[trial_id] + 1
        if (success):
            self.rewards[trial_id] = self.rewards[trial_id] + 1

        # self.metric.append(sum(self.rewards) * 1.0 / sum(self.trials))

    def select_arm(self):
        sampled_theta = []
        for i in range(self.num_options):
            # Construct beta distribution for posterior
            dist = beta(self.prior[i][0] + self.rewards[i],
                        self.prior[i][1] + self.trials[i] - self.rewards[i])

            # Draw sample from beta distribution
            sampled_theta.append(dist.rvs())

        # Return the index of the sample with the largest value
        return sampled_theta.index(max(sampled_theta))


def make_update_obj(path, reward_item, item=None):
    if item is None:
        item = {}
        if len(path) == 1:
            return {'bandit': {path[0]: reward_item}}
    if len(path) > 1:
        new_item = make_update_obj(path[1:], reward_item, item)
        item = {path[0]: new_item}
    else:
        item.update({'bandit': {path[0]: reward_item}})
    return item


def make_baseStructure(combinations):
    baseStructure = {}
    for item in combinations:
        subStructure = baseStructure
        for tier in item[0:-2]:
            subStructure.setdefault(tier, {})
            subStructure = subStructure[tier]
        subStructure.setdefault(item[-2], [])
        subStructure[item[-2]].append(item[-1])
    return baseStructure
