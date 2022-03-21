import pandas as pd

#First step, divide the geners in more columns.
Movies = pd.read_csv('ml-25m/movies.csv', sep=',')

# dropping null value columns to avoid errors
Movies.dropna(inplace=True)

genList = ["Action","Adventure","Animation","Children",
"Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX",
"Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western","(no genres listed)"]

for i in genList:
    Movies[i] = 0

for index, row in Movies.iterrows():
    genrow = Movies.iloc[index]["genres"].split("|")
    for j in genrow:
        Movies.loc[index,j] = 1

Movies = Movies.drop('genres', 1)

""" Second step, merge tag and movies and create columns for each tag """
GenTags = pd.read_csv('ml-25m/genome-tags.csv', sep=',')
GenScor = pd.read_csv('ml-25m/genome-scores.csv', sep=',')


df = Movies.merge(GenScor,on='movieId')
df = df.merge(GenTags, on='tagId')
df =df.pivot_table(index=['movieId','title','Action','Adventure','Animation','Children','Comedy','Crime',
                   'Documentary','Drama','Fantasy','Film-Noir','Horror',
                   'IMAX','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western','(no genres listed)'],
                   columns='tag', values='relevance', fill_value=0).reset_index().rename_axis(None,axis=1)

df.to_csv("NewMovies.csv",index=False)


""" Thirt step, merge newmovies with film rating """
Ratings = pd.read_csv('ml-25m/ratings.csv', sep=',')
NewMovies = pd.read_csv('NewMovies.csv', sep=',')

Ratings = Ratings.groupby(['movieId'])['rating'].mean().reset_index()
#Ratings mean

def myround(x, prec=2, base=.5):
  return round(base * round(float(x)/base),prec)

Ratings["rating"] = Ratings["rating"].apply(lambda x:myround(x))
#Rounded to the nearest 0.5

df = Ratings.merge(NewMovies,on='movieId')

#delete columns ID and title
df = df.drop('movieId', 1)
df = df.drop('title', 1)

df.dropna(inplace=True)

df.to_csv("DatasetMovies.csv",index=False)