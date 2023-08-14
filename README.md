# Detecting walking in the Parkinson's disease population

## About this project

Human activity recogntion (HAR) models are useful for monitoring the activity of individuals, while wearing smart watches.
One popular model for doing so in the OxWearables group is the Biobank Accelerometer Analysis tool, found [here](https://github.com/OxWearables/biobankAccelerometerAnalysis).

In the Walmsley iteration of the implementation of this tool, the raw accelerometery is predicted into 4 possible labels, sleep, sedentary, light and MVPA, using a Balanced Random Forest model, with hidden markov model (HMM) smoothing, applied to signal related features extracted from 30 second windows of acclerometery.
The balanced random forest, and HMM are trained using a smaller, labelled dataset, [Capture-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001), taken from monitoring 151 healthy participants for 1 day.
This approach to the development of HMM models has shown decent performance in identifying windows of high and low activity, capable of further epidemiology analysis.

Building on this work, OxWearables developed the stepcount tool, found [here](https://github.com/OxWearables/stepcount).
This package took a similar approach of training using a smaller dataset to be applied on the large UK Biobank population.
In this case, the stepcount model first uses with a balanced random forest, or self supervised Resnet 18 model with HMM smoothing, to detect periods of likely walking, and then uses a peak counter to count steps.
This model is trained on the [OxWalk](https://ora.ox.ac.uk/objects/uuid:19d3cb34-e2b3-4177-91b6-1bad0e0163e7) dataset, collected from 39 healthy participants monitored for approximately 1 hour.

The identification of periods of activity/walking can be very useful for the identification of health conditions of individuals.
Particiularly focused on walking, if we are able to reliably identify when participants are walking, this can give incite into properties like:

* Walking incidence in the average day
* Length of sustained bouts of walking
* Intensity/speed of walking

For Parkinson's disease, various walking related properties can be strong indicators for disease condition.
The early monitoring of walking properties can therefore give insight into properties such as:

* The progression of the disease from early to late stages
* The symptomatic walking behaviour of those with the disease
* The efficacy of treatments addressing the walking behaviours of those with the disease

However, as we know that PD alters motor conditions of participants, therefore we cannot be certain that a walking model trained on only the healthy population, can seemlessly applied to the PD population and generate reliable predictions.
Therefore this repository has been developed to assess the performance of the walking classifiers the PD population, and attempt to generate new classifiers capable of improving this performance.

To do so, we must use a walking labelled dataset of wrist worn accelerometery, monitoring a PD population. This is the [MJFF Levodopa Response dataset](https://doi.org/10.7303/syn20681023), that monitored 28 PD participants over 4 days.
For 2 days in clinic, 1 of which was off all medication, and 2 days at home, a battery of tasks were given to participants to perform, some of which included walking.
It should be noted that as this dataset was collected based on a battery of tasks rather than in free living, HAR models will be evaluated without the HMM, that is intended to learn natural patterns in transitions between activities in free living.  

## Getting started

This repo can either be run from command line, or using jupyter notebooks.
Before this however, we must ensure that all relevant packages are installed:

```bash
# create conda environment
$ conda env create -f env.yml
# activate environment
$ conda activate pd_walk 
```
