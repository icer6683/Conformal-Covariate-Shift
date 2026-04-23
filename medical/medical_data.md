
# Sepsis experiment data: NaCl target setting

This dataset stores the ICU trajectory experiment for a sepsis cohort with **NaCl 0.9% as the prediction target**.

---

# Cohort definition

Patients are included if:

- they are diagnosed with **sepsis** using one of the selected ICD codes
- each of the three CHART covariates is **not all-zero across the 24 hourly time stamps**
- the **NaCl 0.9% trajectory** is **not all-zero across the 24 hourly time stamps**

The three **CHART covariates** are:

- Heart Rate  
- Respiratory Rate  
- O2 saturation pulseoxymetry  

The **target variable** is:

- **NaCl 0.9% (target)**

The **static covariates** included for each patient are:

- Age
- gender
- ethnicity

The **split variable** is:

- Norepinephrine

---

# Imputation rule

For the three CHART covariates:

- If the trajectory is **all zeros**, the patient is excluded.
- Otherwise, **any zero entries are imputed** using that patient's **median of the nonzero values** of that trajectory.

For the NaCl 0.9% target:

- If the trajectory is **all zeros**, the patient is excluded.
- Otherwise, **zeros are left unchanged**.

This keeps NaCl sparse while ensuring physiologic signals remain usable.

---

# TrainCal / Test split

Patients are divided using **Norepinephrine exposure during the first 12 hourly time stamps (t0..t11)**.

**TrainCal set**
- patients with **no Norepinephrine usage** in any of the **first 12 hourly time stamps**.
- Note: a TrainCal patient *may* have nonzero Norepinephrine later in the trajectory (t12..t23); only the first 12 hours are inspected.

**Test set**
- patients with **any nonzero Norepinephrine usage** in the **first 12 hourly time stamps**.

Norepinephrine itself is **not used as a predictive covariate**, but only to induce a distribution shift.

**Resulting cohort sizes** (verified on the current `.pkl`):

| Set      | n patients |
|----------|-----------:|
| TrainCal |      9264  |
| Test     |      5827  |
| Total    |     15091  |

Compared with the previous rule (any Norepinephrine anywhere in t0..t23 → Test), 664 patients who receive Norepinephrine only in hours 12..23 are reclassified from Test to TrainCal. The shift is now grounded on early-ICU vasopressor initiation rather than any-time exposure.

---

# Saved pickle structure

The `.pkl` file stores a dictionary with four entries:

- `patient_ids_traincal`
- `patient_trajectory_list_traincal`
- `patient_ids_test`
- `patient_trajectory_list_test`

The `patient_ids_*` objects are **lists of folder IDs**.

The `patient_trajectory_list_*` objects are **lists with the same order as the patient IDs**.

Each element of the trajectory list is a **patient dictionary**.

---

# Patient dictionary structure

Dynamic variables are stored as pandas DataFrames with columns:

- `hour`
- `value`

Static variables are stored as **scalar values**.

A typical patient dictionary contains:

- `Age`
- `gender`
- `ethnicity`
- `Heart Rate`
- `Respiratory Rate`
- `O2 saturation pulseoxymetry`
- `NaCl 0.9% (target)`
- `Norepinephrine`

Example structure:

```
patient_dict = {
    "Age": 46,
    "gender": "F",
    "ethnicity": "WHITE",
    "Heart Rate": DataFrame(24×2),
    "Respiratory Rate": DataFrame(24×2),
    "O2 saturation pulseoxymetry": DataFrame(24×2),
    "NaCl 0.9% (target)": DataFrame(24×2),
    "Norepinephrine": DataFrame(24×2)
}
```

---

# Example usage

## Load the dataset

```python
import pickle

with open("sepsis_experiment_data_nacl_target.pkl", "rb") as f:
    data = pickle.load(f)

patient_ids_traincal = data["patient_ids_traincal"]
patient_trajectory_list_traincal = data["patient_trajectory_list_traincal"]

patient_ids_test = data["patient_ids_test"]
patient_trajectory_list_test = data["patient_trajectory_list_test"]
```

---

## Inspect one TrainCal patient

```python
i = 0

pid = patient_ids_traincal[i]
patient = patient_trajectory_list_traincal[i]

print(pid)
print(patient["Age"])
print(patient["gender"])
print(patient["ethnicity"])

print(patient["Heart Rate"].head())
print(patient["NaCl 0.9% (target)"].head())
```

---

## Inspect one Test patient

```python
i = 0

pid = patient_ids_test[i]
patient = patient_trajectory_list_test[i]

print(pid)
print(patient["Norepinephrine"].head())
```

---

## Convert one patient to a trajectory table

```python
import pandas as pd

def show_patient(patient_dict):
    dynamic_keys = [
        "Heart Rate",
        "Respiratory Rate",
        "O2 saturation pulseoxymetry",
        "NaCl 0.9% (target)",
        "Norepinephrine"
    ]

    df_patient = pd.DataFrame({
        k: patient_dict[k]["value"].values for k in dynamic_keys
    }).T

    df_patient.columns = [f"t{j}" for j in range(df_patient.shape[1])]
    return df_patient

df_patient = show_patient(patient_trajectory_list_traincal[0])
print(df_patient)
```

---

# Notes

- The **target variable is a medication trajectory**, not a physiologic CHART measurement.
- Static variables are stored directly in each patient dictionary for convenience.
- **Norepinephrine is stored only to define the dataset split** and should not be used as a model covariate.
