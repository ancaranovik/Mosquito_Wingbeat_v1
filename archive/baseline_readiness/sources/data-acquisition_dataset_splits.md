> ## Documentation Index
> Fetch the complete documentation index at: https://docs.edgeimpulse.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Define dataset split

> Define training, validation, and test splits with ratios, stratification, and metadata-based grouping controls.

Advanced control over your dataset splits is provided through the **Define dataset split** functionality. This includes specifying split percentages for your training, validation, and test sets, stratifying your data by labels and/or metadata, and grouping your data by metadata values to prevent data leakage.

<Frame caption="Define dataset split modal">
  <img src="https://mintcdn.com/edgeimpulse/2qJe3D9U1mWiFqsd/.assets/images/data-acquisition-dataset-define-split-modal.png?fit=max&auto=format&n=2qJe3D9U1mWiFqsd&q=85&s=2998c5e94e8c6e00ca5dbb412a679668" alt="Define dataset split modal with split ratios, stratification, and grouping controls" width="1531" height="1000" data-path=".assets/images/data-acquisition-dataset-define-split-modal.png" />
</Frame>

To access the dataset split functionality, go to your project and navigate to the data acquisition page. On the dataset tab, click the **⋮** in the top right of the dataset pane, and then click on **Define dataset split** menu item.

<Frame caption="Define dataset split menu item">
  <img src="https://mintcdn.com/edgeimpulse/13vUiXg7-2jRB1VY/.assets/images/data-acquisition-dataset-define-split-menu.png?fit=max&auto=format&n=13vUiXg7-2jRB1VY&q=85&s=48797557d58f6730c884888cd1c86b28" alt="Dataset pane menu with the Define dataset split action highlighted" width="1531" height="1000" data-path=".assets/images/data-acquisition-dataset-define-split-menu.png" />
</Frame>

## Split ratios

<Info>
  **Validation set only available when explicit validation set setting is enabled**

  The validation set option is only available when the explicit validation set [advanced setting](/studio/projects/data-acquisition/dataset/advanced-settings) is enabled. If this setting is disabled, no separate validation set option will be shown.
</Info>

Split ratios define the percentage of data assigned to each split (training, validation, and test). These values are a percentage of the total dataset, and must sum to 100%.

These are target ratios. Other constraints, such as grouping by metadata, may cause the final result to deviate from them.

## Stratified splitting

Stratified splitting attempts to preserve the label distribution across splits. Each split will contain similar proportions of each label, as much as possible.

This is useful when labels are imbalanced and you want each dataset split to reflect the overall distribution. Disabled samples can optionally be excluded.

Stratification is best-effort and may not be achievable for labels with few data samples.

## Metadata split constraints

Metadata-based constraints give you more control over how samples are distributed between splits. These operate alongside label stratification and may interact with it.

There are two types of constraints: stratify by metadata (a soft constraint) and group by metadata (a hard constraint).

### Stratify splits by metadata

Stratifying the splits by metadata attempts to preserve the distribution of selected metadata keys across splits, similar to label stratification. For example, if you stratify by a `device_type` key, each split will aim to contain a similar proportion of samples from each device value.

This applies in addition to label stratification and is a best-effort only; it may not hold for highly specific combinations of labels and metadata.

### Group samples by metadata

Grouping by metadata ensures that all samples sharing the same metadata value for a specific key are placed in the same split. For example, keeping all samples from a single subject, device, or session together. This is commonly used to prevent data leakage.

This is a hard constraint. Entire groups are assigned together, even if this causes imbalance or overrides stratification behaviour.

## Data preview

Click the `Preview dataset split` button to simulate the split before applying it. This helps you validate that your constraints produce the expected outcomes.

There are three graph views available, each with a corresponding table: overall, assignment, and stratification.

### Overall view

The overall view shows the final distribution of samples across splits. It is the simplest summary of the result, helps confirm that the splits roughly match the intended ratios, and that no split is unexpectedly small or empty.

<Frame caption="Data preview - overall view graph">
  <img src="https://mintcdn.com/edgeimpulse/13vUiXg7-2jRB1VY/.assets/images/data-acquisition-dataset-split-preview-overall-graph.png?fit=max&auto=format&n=13vUiXg7-2jRB1VY&q=85&s=34466728209c1d2538006f1eeca4ceed" alt="Overall data preview graph showing split distribution before reassignment" width="905" height="590" data-path=".assets/images/data-acquisition-dataset-split-preview-overall-graph.png" />
</Frame>

<br />

<Frame caption="Data preview - overall view table">
  <img src="https://mintcdn.com/edgeimpulse/13vUiXg7-2jRB1VY/.assets/images/data-acquisition-dataset-split-preview-overall-table.png?fit=max&auto=format&n=13vUiXg7-2jRB1VY&q=85&s=544b2ac79db895cc3a0d516728459aa7" alt="Overall data preview table listing sample counts by label and split" width="905" height="221" data-path=".assets/images/data-acquisition-dataset-split-preview-overall-table.png" />
</Frame>

### Assignment view

The assignment view shows how samples are grouped into buckets and assigned to splits.

A bucket represents the unit of assignment. When stratifying by metadata, buckets contain the values of the selected targets, for example `hey_arduino|device_a|session_2`.

When grouping is enabled, buckets correspond to those grouping constraints. Because grouped buckets must stay together, they appear as 100% assigned to a single split.

<Frame caption="Data preview - assignment view graph">
  <img src="https://mintcdn.com/edgeimpulse/13vUiXg7-2jRB1VY/.assets/images/data-acquisition-dataset-split-preview-assignment-graph.png?fit=max&auto=format&n=13vUiXg7-2jRB1VY&q=85&s=b358014aea7684414ebb82d396db0986" alt="Assignment view graph showing selected samples assigned to dataset splits" width="905" height="590" data-path=".assets/images/data-acquisition-dataset-split-preview-assignment-graph.png" />
</Frame>

<br />

<Frame caption="Data preview - assignment view table">
  <img src="https://mintcdn.com/edgeimpulse/13vUiXg7-2jRB1VY/.assets/images/data-acquisition-dataset-split-preview-assignment-table.png?fit=max&auto=format&n=13vUiXg7-2jRB1VY&q=85&s=c43e852f96a9a7a62cf803ec806df7a5" alt="Assignment view table listing dataset split assignments for selected samples" width="905" height="365" data-path=".assets/images/data-acquisition-dataset-split-preview-assignment-table.png" />
</Frame>

### Stratification view

<Info>
  **Only available when stratification is enabled**

  The stratification view is only available when stratification by label or metadata is enabled. If stratification is disabled, this view will not appear in the dropdown selector for the different graph views.
</Info>

The stratification view evaluates how well balance targets were achieved.

It looks at one balancing target at a time, such as a label or metadata key, and shows how its values are distributed across splits. This makes it easier to spot imbalance, especially when grouping constraints are applied.

Use the dropdown at the top of the graph to switch between stratification per label or a specific metadata key. This dropdown appears only when the stratification view is selected.

<Frame caption="Data preview - stratification view graph">
  <img src="https://mintcdn.com/edgeimpulse/13vUiXg7-2jRB1VY/.assets/images/data-acquisition-dataset-split-preview-stratification-graph.png?fit=max&auto=format&n=13vUiXg7-2jRB1VY&q=85&s=fa02eb4f67b8b41b25dfe78114f73919" alt="Stratification view graph comparing label or metadata balance across splits" width="905" height="590" data-path=".assets/images/data-acquisition-dataset-split-preview-stratification-graph.png" />
</Frame>

<br />

<Frame caption="Data preview - stratification view table">
  <img src="https://mintcdn.com/edgeimpulse/13vUiXg7-2jRB1VY/.assets/images/data-acquisition-dataset-split-preview-stratification-table.png?fit=max&auto=format&n=13vUiXg7-2jRB1VY&q=85&s=3ccb84aac22fbe4afd0b906d5d1dedfe" alt="Stratification view table showing stratified counts across dataset splits" width="905" height="390" data-path=".assets/images/data-acquisition-dataset-split-preview-stratification-table.png" />
</Frame>

## Create dataset split

Click `Create dataset split` to apply the configuration. This immediately reassigns samples between splits, enforcing grouping constraints and applying stratification where possible.

## Additional resources

* [Data acquisition](/studio/projects/data-acquisition)
* [Dataset](/studio/projects/data-acquisition/dataset)
* [Metadata](/studio/projects/data-acquisition/dataset/metadata)
