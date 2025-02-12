import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import minimize

from codes.worker import ByzantineWorker, repeat_model
from codes.aggregator import DecentralizedAggregator, clip


class DecentralizedByzantineWorker(ByzantineWorker):
    def __init__(self, target=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The target of attack
        self.target = target
        self.tagg = None
        self.target_good_neighbors = None

    def _initialize_target(self):
        if isinstance(self.target, int):
            nodes_set = set(self.running["neighbor_workers"])
            to_rm = set()
            while isinstance(self.target, int) or len(nodes_set) == 0:
                to_add = set()
                for w in nodes_set:
                    if w.index == self.target:
                        self.target = w
                        self.tagg = w.running["aggregator"]
                        self.target_good_neighbors = self.simulator.get_good_neighbor_workers(
                            w.running["node"]
                            )
                        break
                    to_rm.add(w)
                    to_add.update(w.running["neighbor_workers"])
                for idx_to_rm in to_rm:
                    nodes_set.discard(idx_to_rm)
                nodes_set.update(to_add)

        if self.target is None or isinstance(self.target, int):
            assert len(self.running["neighbor_workers"]) >= 1
            self.target = self.running["neighbor_workers"][0]
            self.tagg = self.target.running["aggregator"]
            self.target_good_neighbors = self.simulator.get_good_neighbor_workers(
                self.target.running["node"]
            )


class DissensusWorker(DecentralizedByzantineWorker):
    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def _attack_decentralized_aggregator(self, mixing=None):
        tm = self.target.running["flattened_models"][self.target.index]

        # Compute Byzantine weights
        partial_sum = []
        partial_byz_weights = []
        for neighbor in self.target.running["neighbor_workers"]:
            nm = neighbor.running["flattened_models"][neighbor.index]
            nn = neighbor.running["node"]
            nw = mixing or self.tagg.weights[nn.index]
            if isinstance(neighbor, ByzantineWorker):
                partial_byz_weights.append(nw)
            else:
                partial_sum.append(nw * (nm - tm))

        partial_sum = sum(partial_sum)
        partial_byz_weights = sum(partial_byz_weights)

        return tm, partial_sum / partial_byz_weights

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            # Dissensus using the gossip weight
            tm, v = self._attack_decentralized_aggregator()
            self.running["flattened_modes"] = repeat_model(self, tm - self.epsilon * v)
        else:
            # TODO: check
            # Dissensus using the gossip weight
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            tm, v = self._attack_decentralized_aggregator(mixing)
            self.running["flattened_models"] = repeat_model(self, tm - self.epsilon * v)



##################################################################################

class EchoWorker(DecentralizedByzantineWorker):
    def __init__(self, targeted, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targeted = targeted

    def _attack_decentralized_aggregator(self, mixing=None):
        raise NotImplementedError
        # TODO: same as below but clip stuff

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            self.running["flattened_models"] = self._attack_decentralized_aggregator()
        else:
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            self.running["flattened_models"] = self._attack_decentralized_aggregator(mixing)

class EchoNoClipWorker(DecentralizedByzantineWorker):
    def __init__(self, targeted, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targeted = targeted

    def _attack_decentralized_aggregator(self, mixing=None):
        thetas = {}
        if self.targeted:
            tm = self.target.running["flattened_models"][self.target.index]
            adv_scale = mixing or self.tagg.weights[self.index]
            tm /= adv_scale
            for w in self.running["neighbor_workers"] + [self]:
                thetas[w.index] = tm
        else:
            for w in self.running["neighbor_workers"]:
                tm = w.running["flattened_models"][w.index]
                adv_scale = mixing or w.running["aggregator"].weights[self.index]
                thetas[w.index] = tm/adv_scale
            # TODO: below is stupid trick to have a self model (that we never use) because + self in loop does not work.
            thetas[self.index] = tm/adv_scale
        return thetas

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            self.running["flattened_models"] = self._attack_decentralized_aggregator()
        else:
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            self.running["flattened_models"] = self._attack_decentralized_aggregator(mixing)

class SandTrapWorker(DecentralizedByzantineWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _attack_decentralized_aggregator(self, mixing=None):
        raise NotImplementedError
        # TODO: do stuff with clipping
        "network_contrib += (v + clip(data[j-self.b][t] - v, tau))*m[j]\n",
        "def f(vv):\n",
        "   return abs(network_contrib + (v + clip(vv - v, tau))*m[byzantine_node])\n",
        "inv_network_contrib = minimize(f, -network_contrib, options={'maxiter':10}).x[0]\n",

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            self.running["flattened_models"] = self._attack_decentralized_aggregator()
        else:
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            self.running["flattened_models"] = self._attack_decentralized_aggregator(mixing)


class SandTrapNoClipWorker(DecentralizedByzantineWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _attack_decentralized_aggregator(self, mixing=None):
        thetas = {}
        tm = self.target.running["flattened_models"][self.target.index]
        network_contrib = 0
        adv_scale = 0 #rescaling process as the adv_weights are not in the loop
        for w in self.target.running["neighbor_workers"]:
            nw = mixing or self.tagg.weights[w.index]
            if isinstance(w, ByzantineWorker):
                adv_scale += nw
            else:
                network_contrib += w.running["flattened_models"][self.target.index] * nw
        network_contrib /= adv_scale
        for w in self.running["neighbor_workers"] + [self]:
            if w.index == self.target.index:
                thetas[w.index] = -network_contrib
            else:
                thetas[w.index] = network_contrib
        return thetas

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            self.running["flattened_models"] = self._attack_decentralized_aggregator()
        else:
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            self.running["flattened_models"] = self._attack_decentralized_aggregator(mixing)


class StateOverrideWorker(DecentralizedByzantineWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_state = torch.zeros(1199882)
        # TODO: not hardcode this number
        self.target_state = self.target_state.to(self.device)

    def _attack_decentralized_aggregator(self, mixing=None):
        thetas = {}
        for w in self.running["neighbor_workers"] + [self]:
            network_contrib = 0
            adv_scale = 0 #rescaling theta_i as the advs' weights are "lost"
            tm = w.running["flattened_models"][w.index]
            for ww in w.running["neighbor_workers"]:
                nw = mixing or w.running["aggregator"].weights[ww.index]
                if isinstance(ww, ByzantineWorker):
                    adv_scale += nw
                else:
                    tau = w.running["aggregator"].tau
                    network_contrib += (tm + clip(ww.running["flattened_models"][w.index] - tm, tau)) * nw

            def f(x):
               return torch.norm(self.target_state.cpu() - (network_contrib.cpu() + (tm.cpu() + clip(torch.from_numpy(x) - tm.cpu(), tau)) * w.running["aggregator"].weights[self.index])).item()
            x = minimize(f, (self.target_state - network_contrib).cpu(), options={'maxiter':10})
            print(x)
            thetas[w.index] = x
        return thetas

         # TODO: stuff with clipping
        "network_contrib += (v + clip(data[j-self.b][i] - v, tau))*m[j]\n",
        "def f(vv):\n",
        "   return abs(network_contrib + (v + clip(vv - v, compute_tau(data, self.graph, i)))*m[byzantine_node] - self.target)\n",
        "vv = minimize(f, self.target -network_contrib, options={'maxiter':10}).x[0]\n",

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            v = self._attack_decentralized_aggregator()
        else:
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            v = self._attack_decentralized_aggregator(mixing)
        self.running["flattened_models"] = v



class StateOverrideNoClipWorker(DecentralizedByzantineWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_state = torch.zeros(1199882)
        # TODO: not hardcode this number
        self.target_state = self.target_state.to(self.device)

    def _attack_decentralized_aggregator(self, mixing=None):
        thetas = {}
        for w in self.running["neighbor_workers"] + [self]:
            network_contrib = 0
            adv_scale = 0 #rescaling theta_i as the advs' weights are "lost"
            for ww in w.running["neighbor_workers"]:
                nw = mixing or w.running["aggregator"].weights[ww.index]
                if isinstance(ww, ByzantineWorker):
                    adv_scale += nw
                else:
                    network_contrib += ww.running["flattened_models"][w.index] * nw
            thetas[w.index] = (self.target_state - network_contrib)/adv_scale
        return thetas


    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        if isinstance(self.tagg, DecentralizedAggregator):
            v = self._attack_decentralized_aggregator()
        else:
            mixing = 1 / (len(self.target.running["neighbor_workers"]) + 1)
            v = self._attack_decentralized_aggregator(mixing)
        self.running["flattened_models"] = v


# TODO: add tm - self.epsilon * v !

##################################################################################



class BitFlippingWorker(ByzantineWorker):
    def __str__(self) -> str:
        return "BitFlippingWorker"

    def pre_aggr(self, epoch, batch):
        self.running["flattened_models"] = repeat_model(self, -self.running["flattened_model"][self.index])


class LabelFlippingWorker(ByzantineWorker):
    def __init__(self, revertible_label_transformer, *args, **kwargs):
        """
        Args:
            revertible_label_transformer (callable):
                E.g. lambda label: 9 - label
        """
        super().__init__(*args, **kwargs)
        self.revertible_label_transformer = revertible_label_transformer

    def train_epoch_start(self) -> None:
        super().train_epoch_start()
        self.running["train_loader_iterator"].__next__ = self._wrap_iterator(
            self.running["train_loader_iterator"].__next__
        )

    def _wrap_iterator(self, func):
        def wrapper():
            data, target = func()
            return data, self.revertible_label_transformer(target)

        return wrapper

    def _wrap_metric(self, func):
        def wrapper(output, target):
            return func(output, self.revertible_label_transformer(target))

        return wrapper

    def add_metric(self, name, callback):
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = self._wrap_metric(callback)

    def __str__(self) -> str:
        return "LabelFlippingWorker"


class ALittleIsEnoughAttack(DecentralizedByzantineWorker):
    """
    Adapted for the decentralized environment.

    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
    """

    def __init__(self, n, m, z=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Number of supporters
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
        self.n_good = n - m

    def get_gradient(self):
        return 0

    def set_gradient(self, gradient):
        pass

    def apply_gradient(self):
        pass

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        tm = self.target.running["flattened_models"][self.target.index]
        models = [tm]
        for neighbor in self.target_good_neighbors:
            models.append(neighbor.running["flattened_models"][neighbor.index])

        stacked_models = torch.stack(models, 1)
        mu = torch.mean(stacked_models, 1)
        std = torch.std(stacked_models, 1)

        self.running["flattened_models"] = repeat_model(self, mu - std * self.z_max)


class IPMAttack(DecentralizedByzantineWorker):
    def __init__(self, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def get_gradient(self):
        return 0

    def set_gradient(self, gradient):
        pass

    def apply_gradient(self):
        pass

    def pre_aggr(self, epoch, batch):
        self._initialize_target()

        tm = self.target.running["flattened_models"][self.target.index]
        models = [tm]
        for neighbor in self.target_good_neighbors:
            models.append(neighbor.running["flattened_models"][neighbor.index])

        self.running["flattened_models"] = repeat_model(self, -self.epsilon * sum(models) / len(models))


def get_attackers(
    args, rank, trainer, model, opt, loss_func, loader, device, lr_scheduler
):
    if args.attack == "BF":
        return BitFlippingWorker(
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )

    if args.attack == "LF":
        return LabelFlippingWorker(
            revertible_label_transformer=lambda label: 9 - label,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )

    if args.attack.startswith("ALIE"):
        if args.attack == "ALIE":
            z = None
        else:
            z = float(args.attack[4:])
        attacker = ALittleIsEnoughAttack(
            n=args.n,
            m=args.f,
            z=z,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack == "IPM":
        attacker = IPMAttack(
            epsilon=0.1,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack.startswith("sandtrap"):
        target = None
        if len(args.attack) > len("sandtrap"):
            target = int(args.attack[len("sandtrap") :])
        attacker = SandTrapNoClipWorker(
            target=target,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack.startswith("stateoverride"):
        # TODO: what do we put as state? All 0 tensor flow?
        attacker = StateOverrideNoClipWorker(
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack.startswith("scc_stateoverride"):
        attacker = StateOverrideWorker(
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack.startswith("echo"):
        targeted = False
        target = None
        if len(args.attack) > len("echo"):
            targeted = True
            if int(args.attack[len("echo") :]) != -1:
                target = int(args.attack[len("echo") :])
        attacker = EchoNoClipWorker(
            targeted=targeted,
            target=target,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker

    if args.attack.startswith("dissensus"):
        epsilon = float(args.attack[len("dissensus") :])
        attacker = DissensusWorker(
            epsilon=epsilon,
            simulator=trainer,
            index=rank,
            data_loader=loader,
            model=model,
            loss_func=loss_func,
            device=device,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
        return attacker
    raise NotImplementedError(f"No such attack {args.attack}")
