> ## Documentation Index
> Fetch the complete documentation index at: https://docs.edgeimpulse.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Advanced dataset settings

> Configure explicit validation sets, disable reset behavior, and adjust dataset settings from the Advanced settings modal.

The **Advanced settings** found on the dataset tab provide additional configuration options for you to control and customize the data used in your project. The sections that follow are an overview of the advanced settings available.

<Frame caption="Advanced settings modal">
  <img src="https://mintcdn.com/edgeimpulse/2qJe3D9U1mWiFqsd/.assets/images/data-acquisition-dataset-advanced-settings-modal.png?fit=max&auto=format&n=2qJe3D9U1mWiFqsd&q=85&s=2f0d796649ba42e45d84214a4b59c31d" alt="Advanced settings modal with explicit validation set and reset dataset options" width="1531" height="1000" data-path=".assets/images/data-acquisition-dataset-advanced-settings-modal.png" />
</Frame>

To access the advanced settings, go to your project and navigate to the data acquisition page. On the dataset tab, click the **⋮** in the top right of the dataset pane, and then click on **Advanced settings** menu item. This will open a modal with the settings available.

<Frame caption="Advanced settings menu item">
  <img src="https://mintcdn.com/edgeimpulse/2qJe3D9U1mWiFqsd/.assets/images/data-acquisition-dataset-advanced-settings-menu.png?fit=max&auto=format&n=2qJe3D9U1mWiFqsd&q=85&s=e407bc4c55ac9d74fdd931f813046adb" alt="Dataset pane menu with the Advanced settings action highlighted" width="1531" height="1000" data-path=".assets/images/data-acquisition-dataset-advanced-settings-menu.png" />
</Frame>

## Explicit validation set

By default, Edge Impulse automatically splits your training data into the training and validation sets. Enabling the explicit validation set setting allows you to specify the percentage of data to use for your validation set, explore what samples are contained in the set, and move samples to/from the validation set from/to your training or test sets.

<Frame caption="Validation set samples after enabling explicit validation set">
  <img src="https://mintcdn.com/edgeimpulse/2qJe3D9U1mWiFqsd/.assets/images/data-acquisition-dataset-validation-set.png?fit=max&auto=format&n=2qJe3D9U1mWiFqsd&q=85&s=7c1223fcd826863c4b15335fb0d9938a" alt="Validation set tab showing samples after the explicit validation set is enabled" width="1531" height="1000" data-path=".assets/images/data-acquisition-dataset-validation-set.png" />
</Frame>

This gives you more control over the data used for the validation set, such as ensuring it contains specific samples or has a certain distribution. It also allows you to have a fixed validation set that does not change as you add new data to your training set.

After enabling the explicit validation set setting, a field will appear in the [Define dataset split](/studio/projects/data-acquisition/dataset/splits) modal to specify the percentage of data to use for the validation set.

If you subsequently disable the explicit validation set setting, any samples in your validation set will be moved back to your training set and the validation split percentage field will be removed from the define dataset split modal. Future training jobs will use the standard validation split behaviour.

<Frame caption="Disable explicit validation set warning">
  <img src="https://mintcdn.com/edgeimpulse/2qJe3D9U1mWiFqsd/.assets/images/data-acquisition-dataset-disable-explicit-validation-set-warning.png?fit=max&auto=format&n=2qJe3D9U1mWiFqsd&q=85&s=b37fc2c4ebf2ddd16b7534cff3c4e654" alt="Warning dialog confirming that disabling the explicit validation set will reset splits" width="1531" height="1000" data-path=".assets/images/data-acquisition-dataset-disable-explicit-validation-set-warning.png" />
</Frame>

## Additional resources

* [Data acquisition](/studio/projects/data-acquisition)
* [Dataset](/studio/projects/data-acquisition/dataset)
* [Define dataset split](/studio/projects/data-acquisition/dataset/splits)
