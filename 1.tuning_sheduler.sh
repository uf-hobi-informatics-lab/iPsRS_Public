###### Model Tuning Command: XGBOOST with Random Over Sampling ######
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_fullset_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_fullset_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_fullset_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_fullset_no_age_sex_race.json

sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_contextual_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_contextual_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_contextual_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_contextual_no_age_sex_race.json

sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_individual_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_individual_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_individual_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-ROS-XG/tuning_individual_no_age_sex_race.json


###### Model Tuning Command: Logistic Regression with Random Over Sampling ######

#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_fullset_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_fullset_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_fullset_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_fullset_no_age_sex_race_lr.json

#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_contextual_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_contextual_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_contextual_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_contextual_no_age_sex_race_lr.json

#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_individual_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_individual_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_individual_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-ROS-LR/tuning_individual_no_age_sex_race_lr.json

###############################################################################################
###### Model Tuning Command: XGBOOST with Random Under Sampling ######

sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_fullset_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_fullset_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_fullset_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_fullset_no_age_sex_race.json

sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_contextual_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_contextual_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_contextual_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_contextual_no_age_sex_race.json

sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_individual_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_individual_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_individual_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-RUS-XG/tuning_individual_no_age_sex_race.json

###### Model Tuning Command: Logistic Regression with Random Over Sampling ######

#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_fullset_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_fullset_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_fullset_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_fullset_no_age_sex_race_lr.json

#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_contextual_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_contextual_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_contextual_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_contextual_no_age_sex_race_lr.json

#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_individual_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_individual_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_individual_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-RUS-LR/tuning_individual_no_age_sex_race_lr.json

###############################################################################################
###### Model Tuning Command: XGBOOST with Match ######

sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_fullset_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_fullset_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_fullset_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_fullset_no_age_sex_race.json

sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_contextual_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_contextual_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_contextual_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_contextual_no_age_sex_race.json

sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_individual_full.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_individual_no_age.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_individual_no_age_sex.json
sbatch slurm_model_tuning.sh Settings/Tuning-Match-XG/tuning_individual_no_age_sex_race.json


###### Model Tuning Command: Logistic Regression with Match ######

#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_fullset_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_fullset_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_fullset_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_fullset_no_age_sex_race_lr.json

#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_contextual_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_contextual_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_contextual_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_contextual_no_age_sex_race_lr.json

#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_individual_full_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_individual_no_age_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_individual_no_age_sex_lr.json
#sbatch slurm_model_tuning.sh Settings/Tuning-Match-LR/tuning_individual_no_age_sex_race_lr.json
