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