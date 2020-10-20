# Machine Learning with Amarel using R

Materials for **Machine Learning with Amarel using R** workshop. First held at Rutgers University, Fall 2020.  
<https://github.com/ryandata/AmarelR>

Amarel is the major High Performance Computing (HPC) cluster generally available to Rutgers faculty, students, and staff.  It is named in honor of Dr. Saul Amarel.  See
<https://oarc.rutgers.edu/resources/amarel/> for more information.

The Office of Advanced Research Computing (OARC) supports Amarel and provides their own specialized workshops and training.  See their events and workshops at
<https://it.rutgers.edu/events/category/oarc/?context=oarc> or browse other FAQs and training materials from 
<https://oarc.rutgers.edu>

## Introduction

This workshop introduces some of the details of running on the Amarel cluster, but its focus is a gentle overview of machine learning, using R as the primary tool for analysis.  By working on the Amarel cluster, we use a genuinely "big" dataset as our example, and explore where HPC methods are necessary to handle the data effectively.

## Data

Our primary data example is an 8GB dataset of NYC traffic tickets.  See 
<https://www.kaggle.com/new-york-city/nyc-parking-tickets>
for details.

## Setup for working on Amarel (from off campus)

Two things are required -- **VPN Access** and an **Amarel account**

1. Configure your VPN according to the instructions here:
<https://soc.rutgers.edu/vpn/>\
This will involve installing the Cisco AnyConnect app for your system, activating it, and installing the Duo app for your device for 2-factor authentication.

2. Request your Amarel account here:
<https://oarc.rutgers.edu/resources/amarel/#access>\
If you do not have a PI or advisor (i.e., you are just using and learning Amarel on your own), you can leave the PI field blank.

You will log in to your Amarel account with your usual NetID/password once access is granted. 

You **must** be on the VPN to connect to Amarel.  If you find you cannot log in, this is the first thing to check.  VPN connections also time out, so you sometimes can lose a connection after being logged in.

If you are on the campus network already, you do not need the VPN, but this is rare in 2020!

## Direct terminal access to Amarel

You can access Amarel directly by using ssh to go to <https://amarel.rutgers.edu>\
Even if you plan on using another interface, like OnDemand, it is good to have a terminal connection active in order to monitor and run basic commands.

## SLURM

A key to working on Amarel is familiarizing yourself with **slurm**, which schedules jobs.

<https://slurm.schedmd.com>

Key commands are **srun**, **squeue**, and **sbatch**

See OARC's **Technical User Guide** at 
<https://sites.google.com/view/cluster-user-guide/home> for more information.

## OnDemand

There are additional ways to use Amarel, such as FastX, which you can learn about if you follow OARC's **Weekly Open Workshop Series** at <https://oarc.rutgers.edu/training-and-scientific-consultation/>.

However, **OnDemand** provides the most general-purpose GUI.

Connect to OnDemand (while on the VPN) at
<https://ondemand.hpc.rutgers.edu>

With OnDemand, you can launch an interactive RStudio session, Jupyter Notebook, or other options.  Be sure to request enough time and computing resources for your tasks, but not too much more than you need, otherwise other users will be disadvantaged.

