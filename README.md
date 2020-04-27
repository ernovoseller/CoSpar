# CoSpar: Efficient Online Learning from Human Feedback
This repository contains code for reproducing the simulation results and plots in Section 4 of the following paper:

**Preference-Based Learning for Exoskeleton Gait Optimization**<br/>
Maegan Tucker\*, Ellen Novoseller\*, Claudia Kann, Yanan Sui, Yisong Yue, Joel W. Burdick, and Aaron D. Ames<br/>
IEEE Conference on Robotics and Automation (ICRA), 2020<br/>
(*equal contribution)<br/>
[PDF](https://arxiv.org/abs/1909.12316) &nbsp;&nbsp;&nbsp; [Video](https://www.youtube.com/watch?v=-27sHXsvONE)

The code is divided into two subfolders, corresponding to the compass-gait biped simulations and the 2D synthetic function simulations, respectively. The Python files included are as follows:

1)	Scripts titled Optimize_*.py contain the code for running the simulations.
2)	Scripts titled Plot_*.py generate the plots in Section 4 of the paper.
3)	Scripts titled Animate_*.py generate the image stacks used to make the animations in the [video](https://www.youtube.com/watch?v=-27sHXsvONE) accompanying the paper.
4)	Preference_GP_learning.py and CoSpar_feedback_functions.py contain helper functions called by 1), 2), and 3).
5)	Generate_and_plot_2D_objective_functions.py contains the code for generating the 2D synthetic objective functions used in the 2nd set of simulations.
6)	Illustrate_self_sparring.py makes some plots of posterior samples (for the compass-gait biped) that were used in the [video](https://www.youtube.com/watch?v=-27sHXsvONE) accompanying the paper.

The folders "Compass_gait_biped_simulations/Compass_biped_results/" and "2D function simulations/Sim_results/" contain the simulation results appearing Figures 2-4 in the paper.
