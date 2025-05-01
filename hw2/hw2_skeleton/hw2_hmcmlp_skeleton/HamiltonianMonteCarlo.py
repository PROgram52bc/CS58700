# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 02:10:04 2022

@author: Mir Imtiaz Mostafiz

implements Hamiltonian Monte Carlo algorithm for Bayesian Averaging
"""

import torch 
import numpy as np

import NeuralNetwork as mynn

DEBUG_ACCEPT = False


class HamiltonianMonteCarloSampler:
    
    def __init__(self, nnModel, rng, device):
        
        self.base_model = nnModel
        
        self.model = mynn.NeuralNetwork(nnModel.shape, nnModel.learning_rate).to(device)
        
        self.shapes = [p.shape for p in self.model.parameters()]
        
        self.rng = rng
        pass

    def __repr__(self):
        
        ret = str(self.model)
        ret += "shapes" + str(self.shapes)
        return ret
    
    def get_sampled_velocities(self, stddv):
        """
        Sample random velocities from zero-mean Gaussian for all parameters.
        

        Parameters
        ----------
        stddv : float32
            standard deviation for all parameters sampling.

        Returns
        -------
        velocities : list of tensors with the same shape as each shape in self.shape sampled velocities for all parameters.

        """
        """
        TO DO COMPLETE IMPLEMENTATION
        """
        
        raise NotImplementedError("Complete Implementation")
        
    
    def leapfrog(self, velocities, delta, /, *ARGS):
        """
        In-place leapfrog iteration.
        It should update `list(self.model.parameters())` as position $x$ in
        HMC.
        It should update `velocities` as momentum $p$ in HMC.
        

        Parameters
        ----------
        velocities : list of length(self.shapes), float32
            sampled velocities for all parameters.
        delta : float32
            delta in HMC algorithm.
        *ARGS : (X, y, y_1hot) as described in utils.py and NeuralNetwork model learning
            
        Returns
        -------
        velocities : list of length(self.shapes), float32
            leapfrog updated velocities for all parameters.

        """
        
        """
        TO DO COMPLETE IMPLEMENTATION
        """
        
        raise NotImplementedError("Complete Implementation")
        
        #phi half = phi sent - delta half * U grad step
            
        
        #X new = X old + delta * phi half step
        
            
            
        #phi new = phi half - delta half * U grad new step
        
        

    def accept_or_reject(self, potential_energy_previous, potential_energy_current, 
                         kinetic_energy_previous, kinetic_energy_current):
        """
        Given the potential and kinetic energies  of the last sample and new sample, 
        check if we should accept new sample.
        If True, we will accept new sample.
        If False, we will reject new sample, and repeat the last sample.
        

        Parameters
        ----------
        potential_energy_previous : float32
            potential energy of last sample.
        potential_energy_current : float32
            potential energy of new sample.
        kinetic_energy_previous : float32
            kinetic energy of last sample.
        kinetic_energy_current : float32
            kinetic energy of new sample.

        Returns
        -------
        boolean
            True if to accept, False if to reject.

        """
        
        """
        TO DO COMPLETE IMPLEMENTATION
        """
        
        raise NotImplementedError("Complete Implementation")
        
    
    def sample(self, n, std_dev, delta, num_leapfrogs, /, *ARGS):
        """
        Sample from given parameters using Hamiltonian Monte Carlo.
        

        Parameters
        ----------
        n : int
            number of samples to generate.
        std_dev : float32
            standard deviation for sampling velocities.
        delta : float32
            delta in sampling velocities as in ALgorithm.
        num_leapfrogs : int
            number of leapfrog steps to do.
        *ARGS : (X, y, y_1hot) as described in utils.py and NeuralNetwork model learning
            
        Returns
        -------
        samples : list of length (1 + n), comprising of list of samples (model parameters) of length (self.model.shapes)
            initial and generated samples of model parameters.

        """

        # Initialize buffer.
        samples = []
        potentials = []

        # print(ARGS[0].shape)
        # Get initial sample from base model parameters
        inits = [param.data for param in self.base_model.parameters()]

        for (param, init) in zip(self.model.parameters(), inits):
            #
            param.data.copy_(init)

        with torch.no_grad():
            #
            nlf = self.model.energy(*ARGS).item()

        samples.append(
            [
                torch.clone(param.data.cpu())
                for param in self.model.parameters()
            ]
        )
        potentials.append(nlf)

        num_accepts = 0
        for i in range(1, n + 1):
            """
            this is running algorithm 1 for n times
            ke is $K(\Phi)$
            nlf is $U(\Phi)
            """
            # Sample a random velocity.
            # Get corresponding potential and kenetic energies.
            velocities = self.get_sampled_velocities(std_dev)
            potential_energy_previous = potentials[-1]
            kinetic_energy_previous = sum(0.5 * torch.sum(velocity ** 2).item() for velocity in velocities)

            # Update by multiple leapfrog steps to get a new sample.
            for _ in range(num_leapfrogs):
                #
                new_velocities = self.leapfrog(velocities, delta, *ARGS)

            with torch.no_grad():
                #
                potential_energy_current = self.model.energy(*ARGS).item()

            kinetic_energy_current = sum(0.5 * torch.sum(new_velocity ** 2).item() for new_velocity in new_velocities)

            # Metropolis-Hasting rejection sampling.
            accept_new = self.accept_or_reject(potential_energy_previous, potential_energy_current,
                                               kinetic_energy_previous, kinetic_energy_current)
            if accept_new:
                # Accept new samples.
                samples.append(
                    [
                        torch.clone(param.data.cpu())
                        for param in self.model.parameters()
                    ],
                )
                potentials.append(potential_energy_current)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Accept", "{:.6f}".format(potential_energy_current)),
                )
            else:
                # Reject new samples.
                # Need to recover model parameters back to the last sample.
                samples.append(samples[-1])
                for (param, init) in zip(self.model.parameters(), samples[-1]):
                    #
                    param.data.copy_(init)
                potentials.append(potential_energy_previous)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Reject", "{:.6f}".format(potential_energy_previous)),
                )
            num_accepts = num_accepts + int(accept_new)
        print("{:s} {:s}".format("-" * 3, "-" * 6))
        print("- Accept%: {:.1f}%".format(float(num_accepts) * 100 / float(n)))
        return samples
        