# Setting Up TensorFlow and Keras on Amazon AWS EC2

## Overview

This document walks you through how to setup TensorFlow and Keras on
an Amazon AWS EC2 GPU instance.

Note: The GPU instance costs about 65 cents an hour. This will quickly
add up ($16 per day, $109 per week, $468 per month). Make sure you
shut down your instance as soon as you are done. Do not let it sit
idle. You are billed even when it is idle.

See pricing details here <https://aws.amazon.com/ec2/pricing/>.

See section on *Terminating Instances* below to verify that you have
terminated all your instances.

## Login

Go to http://aws.amazon.com/

Click on *Sign in to the Console*

Sign in using user name and password

## Launch Instance

Click on *Compute > EC2*

Click on *INSTANCES > Instances*

Click on *Launch Instance*

## Step 1: Choose an Amazon Machine Image (AMI)

Click on *Community AMIs*

Type `23939849` into search box *Search community AMIs*

Click *Select* on AMI `tensor-flow-keras-h5-by-asim-jalis`

## Step 2: Choose an Instance Type

In *Filter by* choose *GPU instances*

Select *Instance Type* of *g2.2xlarge*

Click on *Next: Configure Instance Details*

Note: If you want a cheaper instance that is not as performant instead
of choosing the GPU instance, choose `t2.micro`.

## Step 3: Configure Instance Details

Keep defaults

Click on *Next: Add Storage*


## Step 4: Add Storage

Keep defaults

Click on *Next: Tag Instance*


## Step 5: Tag Instance

Keep defaults

Click on *Next: Configure Security Group*

## Step 6: Configure Security Group

Keep defaults

Click on *Review and Launch*

## Select Key Pair

In *Select an existing key pair or create a new key pair*  the first
time *Create a new key pair*

**VERY IMPORTANT: Make sure you do not use the `deep-kp` key pair that
was provided for the class. This is insecure because it has been
shared. Instead generate a new key pair and keep it somewhere safe.**

For *Key pair name* give your new key pair a name and remember it. 

Click on *Download Key Pair*.

This will download a PEM file. Save this and use it the way we used
the `deep-kp.pem` file in the class.

Click on *Launch Instances*

## Connecting to Instance

Click on *View Instances*

Select the instance. You will see the *Public DNS* value on the bottom
part of the screen. 

In the Mac or Linux console type `ssh -i SECRET-PEM ubuntu@PUBLIC-DNS`

Replace `SECRET-PEM` with the path to your PEM file. 

Replace `PUBLIC-DNS` with the public DNS that you copied from the
screen. It should look like
`ec2-54-173-218-14.compute-1.amazonaws.com`

## Windows and Putty Instructions

For Windows and Putty you will need to convert the PEM file to a PPK
file first. The instructions for this are on this page.

<http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html>

## Terminating Instances

It is very important to terminate your instances after you are done.
Here is how to do it.

If you are not on the *Instances* page, click on the orange cube logo
on the top left corner, and then click on *EC2 > Instances*.

Select instances whose *Instance State* value is *Running*.

Click on *Actions > Instance State > Terminate*

Make sure you *Terminate* and not *Stop*. *Stop* will just shutdown
the instance. You will still keep getting billed a small fee.

Once you have terminated the instance(s) you are done.


