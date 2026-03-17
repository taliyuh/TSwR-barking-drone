# Barking Drone

The project group consists of Mikołaj Lipiński and Bartłomiej Hryniewski.


## Description

The goal of this project is to create a control algorithm for a swarm of drones for the purpose of livestock herding. The project is based on the paper [Robotic Herding of Farm Animals Using a Network of Barking Aerial Drones](https://www.mdpi.com/2504-446X/6/2/29).

## System

Cow behavior will be modeled using Reynolds' rules of flocking behavior. The drones will operate in two modes: 
1. **Gathering mode:** Drones will navigate alongside a complex polygon surrounding the dispersed animals to push the furthest outlier animals towards the center of the herd.
2. **Driving mode:** The polygon simplifies to a circle in order to push the aggregated herd to the target location.

The core control algorithm will be Sliding Mode Control (SMC), utilizing geometric switching logic. 

## Inputs and Outputs

The inputs to the plant (simulation) will consist of:

- Linear drone velocity $v(t)$, where $v(t) \in [0, V_{max}]$
- Angular steering command $u(t)$, where $|u(t)| \le U_{max}$

The outputs of the plant (which serve as feedback to the controller) will be:

- Drone Cartesian coordinates $d(t) = [x(t), y(t)]$
- Drone heading direction vector $a(t)$
- 2D cow positions (specifically the vertices of the herd's convex hull)
- Centroid of the herd $C_o$

