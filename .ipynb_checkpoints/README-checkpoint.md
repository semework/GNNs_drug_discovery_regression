# IMDB_rating_analysis_primer &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<div style="text-align:center;color:white">
    <h2> Movie ratings data analysis, using IMDB's depot and sklearn regression</h2>
</div>

### *A project to explore IMDB movie ratings data and quickly show what influences movie ratings. Approach:*
1.  download IMDB data and store it in AWS s3 bucket
2.  perform EDA and use only numerical data (to save time, one-hot encoding inflates the process)
3.  do simple ML analysis (RF and variable importance) and show features and their relationship to ratings
4.  provide guidelines for better/more accurate approaches

### About this repo   
#### *A jupyter notebook (```IMDB_rating_analysis_primer.ipynb```) which performs the above procedures and provides figures and tables:*

#### Packages needed:
- sklearn
- matplotlib
- numpy
- pandas



!pip install dataframe_image


#### To avoid the version and warning issues, we install s3fs here.

```inside jupyter notebook
!pip install s3fs
```

### An example output (Feature importance with standard deviations):

<img src="assets/images/numerical_features_var_imp.png" style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;"/> 

## Contributing and Permissions

Please do not directly copy anything without my consent. Feel free to reach out to me at https://www.linkedin.com/in/mulugeta-semework-abebe/ for ways to collaborate or use some components.
