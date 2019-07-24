# Lake Depth Data Analysis

## 0. Get Started

Since the program is under beta development, the GUI/statistical decisions are not finalized. There won’t be `Graphical User Interface` available. 

### 0.1 System Requirement

|      Windows       |        Mac OS         |         Linux          |
| :----------------: | :-------------------: | :--------------------: |
| Windows 7 or above | Max OS 10.10 or above | Kernel 4.0.08 or above |



### 0.2 Software Requirement

A minimum of `Python 3.2` is required to run the script. Any additional required packages will be downloaded via `pip3` tool. So, internet connection is required during the first set-up process.

In order to install all required packages, please launch ==Terminal or Command Window==:

1. Type in :

   ```bash
   cd
   ```

   Then, drag your toolkit folder into the ==Command Window==:![](C:\Users\wzhang77\Documents\GitHub\Random-Data-Visualization\doc\img\01.png)

   Then you shall see:

   ![1563995154033](C:\Users\wzhang77\AppData\Roaming\Typora\typora-user-images\1563995154033.png)

   Hit <kbd>Enter</kbd> to continue.

2. Then, run:

   ```bash
   pip install -r requirements.txt
   ```

   The Python 3 pip tool will launch and install all the required packages

   ![](C:\Users\wzhang77\Documents\GitHub\Random-Data-Visualization\doc\img\02.png)

3. Once finished, you may proceed to run the toolkit.

## 1. Dataset Descriptions

We have obtained the weather condition datasets from [MRCC](https://mrcc.illinois.edu/) website, lake ice coverage and lake depth data from [NOAA GLERL](https://www.glerl.noaa.gov/).



### 1.1 Weather Data

All the weather datasets are saved in `.csv` format, obtained from [MRCC Cli-MATE](https://mrcc.illinois.eduCLIMATE/). 

The following rule was followed to harvest useful datasets:

![](C:\Users\wzhang77\Documents\GitHub\Random-Data-Visualization\doc\img\03.PNG)



#### 1.1.1 File Naming

Data was downloaded and saved as `.csv` comma format. File named as:

```bash
<Start Year> - <End Year> [Location Letter].csv
```

![](C:\Users\wzhang77\Documents\GitHub\Random-Data-Visualization\doc\img\04.PNG)

For example, `1981-1990ord.csv` contains datasets collected from ==January 01, 1981== till ==December 31, 1990== at Chicago O’Hare location.

The location letters are:

| ORD            | UGN           | DUGN           |
| -------------- | ------------- | -------------- |
| Coop ID 111549 | WBAN ID 14880 | Coop ID 119029 |



#### 1.1.2 Variables

When it comes to the variables inside of the `.csv` file, there are 8 variables being retrieved for analytic purposes.

![](C:\Users\wzhang77\Documents\GitHub\Random-Data-Visualization\doc\img\05.PNG)

| Name           | Description                                   | Unit |
| -------------- | --------------------------------------------- | ---- |
| Temp           | The ambient temperature at the location       | F    |
| Dewpt          | The dew point temperature at the location     | F    |
| Wind Spd       | The wind speed at the location                | mph  |
| Wind Direction | The wind direction at the location            | deg  |
| Peak Wind Gust | The peak wind gust motion at the location     | mph  |
| Atm Press      | The atmosphere pressure at the location       | hPa  |
| Sea Lev Press  | The sea level pressure at the location        | hPa  |
| Precip         | The precipitation at the location of the hour | in   |



### 1.2 Lake Data

The lake depth and winter ice coverage data are downloaded from the [NOAA server](https://www.glerl.noaa.gov//data/ice/#historical) as `.csv` file.



#### 1.2.1 Ice Coverage Data

As for this project, we are thoroughly focusing on the ice coverage over Michigan Lake area. The data is updated via [NOAA Server](https://www.glerl.noaa.gov/data/ice/daily/mic.csv). The dataset is named as: <kbd>mic.csv</kbd>.

![](C:\Users\wzhang77\Documents\GitHub\Random-Data-Visualization\doc\img\07.PNG)

As we can see here, it doesn’t share a same layout as the weather datasets. The 1^st^ column contains the range of dates which data were being collected. It starts from ==November 14== of current year till ==May 31== of next year.



#### 1.2.2 Lake Depth Data

When it comes to the lake depth, the monthly data can be found on the [NOAA Dashboard](https://www.glerl.noaa.gov/data/dashboard/data/levels/1918_PRES/). The following files are being actively monitored:

![](C:\Users\wzhang77\Documents\GitHub\Random-Data-Visualization\doc\img\06.png)

These datasets are being updated frequently, yet, not daily together. Hence, some pre-processing is needed, which you can find out more in later sections. 