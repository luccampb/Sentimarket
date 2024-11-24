from datetime import datetime
from flask import Flask, render_template, request, redirect, flash
from flask_caching import Cache
from db_schema import db, Users, Company, UserFollow, LastUpdated
from flask_login import LoginManager, login_user, UserMixin, login_required, current_user, logout_user
from flask_mail import Mail, Message
from werkzeug import security
from werkzeug.utils import secure_filename
import requests
import numpy as np
from sqlalchemy import func
import os
from itertools import islice
import matplotlib.pyplot as plt

import urllib.request, json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Tensorflow with keras is used to implement the (stacked) LSTM model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM

#create Flask app instance and initialise database/login/email details
app = Flask(__name__)
app.secret_key = 'xyz789'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///cswkdb.sqlite'
app.config['CACHE_TYPE'] = 'simple'
login_manager = LoginManager()
db.init_app(app)
# Initialises cache
cache = Cache(app)
# Initialises folder to upload things to as the 'static' folder
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Sets up email functionality
app.config['MAIL_SERVER'] = 'smtp-mail.outlook.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sentimarket@outlook.com'
app.config['MAIL_PASSWORD'] = 'cs261project'
mail = Mail(app)

# Creates DB objects
with app.app_context():
    db.create_all()
# Initialises the login manager for flask_login
login_manager.init_app(app)

#load a user based on id for flask login
@login_manager.user_loader
def load_user(user_id):
    return Users.query.filter_by(id=user_id).first()

#log a user out using flask login
@app.route('/logout')
def logout():
    logout_user()
    return redirect("/")

# Store results of this function for 5 minutess
@cache.memoize(300)
# get news stories for index page, given a list of company tickers
def getNewsFeed(tickerlist):
    NewsData = []
    # Query AlphaVantage for news sentiment for all the companies passed to the function
    for ticker in tickerlist:
        NewsData += requests.get("https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=***&apikey=QCLLY08TISZBMXL9".replace("***", ticker)).json()["feed"]
    # Sort the news stories into chronological order
    NewsData = sorted(NewsData, key=lambda x: int(datetime.strptime(x["time_published"], "%Y%m%dT%H%M%S").timestamp()), reverse=True)
    # Use the titles to determine whether the news story is unique in the list or not
    # Avoids duplicating news stories
    titles = set()
    unique_news = []
    for story in NewsData:
        title = story["title"]
        if title not in titles:
            titles.add(title)
            unique_news.append(story)
    return unique_news

#route for home page
@app.route("/", methods=["GET", "POST"])
def index():
    following = [] # Companies followed by the user
    TopIndustry = None # The most common industry followed by the user
    SuggestedCompanies = None # Companies suggested for the user to follow
    NewsData = [] # Stores the result of an AlphaVantage news call
    # Checks that the user is logged in
    if hasattr(current_user, "email"):
        # Queries the database for the most common industry that the user follows
        TopIndustry = db.session.query(Users.id.label('userid'), Company.industry.label('most_followed_industry'), func.count(UserFollow.companyid).label('industry_follow_count')).join(UserFollow, Users.id == UserFollow.userid).join(Company, UserFollow.companyid == Company.id).filter(Users.id == current_user.id).group_by(Users.id, Company.industry).order_by(func.count(UserFollow.companyid).desc()).limit(1).first()
        # If the user has a top industry then get 5 different companies also in that industry 
        if TopIndustry is None:
            SuggestedCompanies = None
        else:
            TopIndustry = TopIndustry[1]
            SuggestedCompanies = db.session.query(Company.ticker, Company.name).outerjoin(UserFollow, (Company.id == UserFollow.companyid) & (UserFollow.userid == current_user.id)).filter(Company.industry == TopIndustry, UserFollow.userid.is_(None)).limit(5).all()
        # Gets the tuple in the table corresponding to the logged in user
        usrQuery = Users.query.filter_by(email=current_user.email).first()
        # Gets all companies followed by the logged in user
        followQuery = UserFollow.query.filter_by(userid=usrQuery.id).all()
        for row in followQuery:
            compQuery = Company.query.filter_by(id=row.companyid).first()
            following.append([compQuery.name, compQuery.ticker])
        # checks if the user is actually following a company
        if len(following) > 0:
            #only pass tickers into news feed function
            # Returns a sorted list of news stories pertaining to the companies followed by the user
            NewsData = getNewsFeed([i[1] for i in following])
            # Issue can arise if AV does not return any news for some companies, this prevents that
            if len(NewsData) > 0:
                last_updated = LastUpdated.query.filter_by(userid=usrQuery.id).first()
                last_updated.timestamp = int(datetime.strptime(NewsData[0]["time_published"], "%Y%m%dT%H%M%S").timestamp())
                db.session.commit()
    # Render the index page, passing in the below variables
    return render_template("index.html", following=following, TopIndustry=TopIndustry, SuggestedCompanies=SuggestedCompanies, NewsData=NewsData)

#route to send summary email
# Sends a personalised email to the user's stored address with 15 stories from companies that they follow
@app.route("/emailsummary", methods=["GET", "POST"])
def emailsummary():
    flash(f"Email summary sent to {current_user.email}")
    msg = Message(f"Sentimarket | {current_user.name}'s news summary", sender="sentimarket@outlook.com", recipients=[current_user.email])
    stories = getNewsFeed([company.ticker for company in get_followed_companies()])
    if len(stories) > 15:
        stories = stories[:15]
    msg.html = render_template("email.html", stories=stories, name=current_user.name)
    mail.send(msg)
    return redirect("/")

#route for signup page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    return render_template("signup.html")

#create user account (on sign up button click)
@app.route("/createaccount", methods=["GET", "POST"])
def create_account():
    email = request.form.get("email")
    password_hash = security.generate_password_hash(request.form.get("password"))
    name = request.form.get("name")
    #make sure no account for email already exists before creating account
    if Users.query.filter_by(email=email).first() is None:
        # Add the user with details from the form into the db. Give them the default profile picture
        db.session.add(Users(email, password_hash, name, "standardprofilepictureyes.png"))
        db.session.commit()
        flash("Account successfully created")
        # Add their details to the LastUpdated table, with time 0
        active_user = Users.query.filter_by(email=email).first()
        db.session.add(LastUpdated(active_user.id, 0))
        db.session.commit()
        # Pass the User tuple to flask_login's login_user method to log them in
        login_user(active_user)
        return redirect("/")
    else:
        flash("Account already registered using email")
        return redirect("/signup")


#route for login page
@app.route("/login")
def login():
    # check the user is not attempting to access the login page when they are already logged in
    if hasattr(current_user, "email"):
        return redirect("/")
    return render_template("login.html")

#log in user (on log in button click)
@app.route('/loginuser', methods=["GET", "POST"])
def log_user_in():
    email=request.form.get("email")
    password=request.form.get("password")
    active_user = Users.query.filter_by(email=email).first()
    # Check that the password from the form matches the stored hash for that account
    if active_user and security.check_password_hash(active_user.password_hash, password):
        # Pass the User tuple to flask_login's login_user method to log them in
        login_user(active_user)
        return redirect("/")
    else:
        flash("Incorrect login details entered")
        return redirect("/login")
    

@app.route("/company/<ticker>", methods=["GET", "POST"])
def company(ticker):
    follow = "Follow"
    following = []
    # Checks if the user is logged in
    if hasattr(current_user, "email"):
        # Gets the user ID of the current user
        usrQuery = Users.query.filter_by(email=current_user.email).all()
        # Gets the company ID of the company on the page
        compQuery = Company.query.filter_by(ticker=ticker).all()
        # Checks if the user follows the company
        followQuery = UserFollow.query.filter(UserFollow.userid==usrQuery[0].id, UserFollow.companyid==compQuery[0].id).first()
        # If the above query has a result then the user is following the company
        if followQuery is not None:
            follow = "Unfollow"
        # Gets the companies followed by the user - used for the following dropdown in the menu bar
        compQuery = get_followed_companies()
        for comp in compQuery:
            following.append([comp.name, comp.ticker])
    # Get company overview info
    daily_prices, daily_sentiment, days, days2, volatility, name, description, industry, weekly_prices, weeks, monthly_prices, months, news = company_chart(ticker)
    # Prevents issue caused when AV does not return any news for a given ticker
    if daily_sentiment == []:
        avg_sentiment = 0
    else:
        avg_sentiment = round(np.mean(daily_sentiment), 4)
    # Get company prediction info
    last70df, predictions, last100DaysList = predicted_vals(ticker)   
    # Render the company page, passing the below variables to the html 
    return render_template("company.html", daily_prices=daily_prices, daily_sentiment=daily_sentiment, days=days, days2=days2, ticker=ticker, volatility=volatility, name=name, description=description, follow=follow, avg_sentiment=avg_sentiment, industry=industry, following=following, weekly_prices=weekly_prices, weeks=weeks, monthly_prices=monthly_prices, months=months, news=news, last70df=last70df, predictions=predictions, last100DaysList=last100DaysList)

# Store the results of this function for 5 minutes
@cache.memoize(300)
def predicted_vals(ticker):
    if (ticker == "AAPL") or (ticker == "META") or (ticker == "BAC") or (ticker == "JPM") or (ticker == "NVDA") or (ticker == "TSLA"):
        a = 0
        #JSON file with daily time series data (date, daily open, daily high, daily low, daily close, daily volume) of specified company,
        #covering 20 years
        urlString = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=QCLLY08TISZBMXL9"%ticker

        #Will use the close values only for the prediction, hence need to save the dates (important as we are using a time series model)
        #and the close values.

        #Save data to this file
        file = "stock_market_data-%s.csv"%ticker

        #If the data has not been saved, save the data to the file and store it as a Pandas dataframe
        if not os.path.exists(file):
            with urllib.request.urlopen(urlString) as url:
                data = json.loads(url.read().decode())
                data = data["Time Series (Daily)"]
                df = pd.DataFrame(columns = ["Date", "Close"])
                for k,v in data.items():
                    date = datetime.strptime(k, "%Y-%m-%d")
                    dataRow = [date.date(), float(v["4. close"])]
                    df.loc[-1,:] = dataRow
                    df.index = df.index + 1
            print("Data saved to : %s"%file)        
            df.to_csv(file)

        # If the data has already been saved, load it to a Pandas dataframe
        else:
            print("File already exists. Loading data from CSV")
            df = pd.read_csv(file)

        #Sorts the dataframe by date (chronologically)
        df = df.sort_values("Date")

        #Stores close values
        close = df[["Close"]]
        dfClose = close.values

        #Uses MinMaxScaler to scale the close values so that the are between 0 and 1, and reshapes the resulting numpy array as an array with
        #one column (and as many rows as necessary).
        scaler = MinMaxScaler((0, 1))
        dfScaled = scaler.fit_transform(np.array(dfClose).reshape(-1, 1))

        #Splits the scaled close values into a training dataset and testing dataset. 90% of the data is used for training, the rest
        #allocated for testing. 
        trainSize = int(len(dfScaled) * 0.9)
        dfTrain = dfScaled[0:trainSize,:]
        dfTest = dfScaled[trainSize:len(dfScaled),:1]

        #Will use the previous n days to predict the (n+1)st day, where n is the number of previous days. This function creates two numpy
        #arrays to represent this, where each item in X stores n sequential days and each corresponding item in Y stores the following day.
        def createDF(data, n):
            X = []
            Y = []
            for i in range(len(data) - n - 1):
                X.append(data[i:(i + n), 0])
                Y.append(data[i + n, 0])
            return np.array(X), np.array(Y)

        #Will use 50 previous days to predict the next day
        numberOfPreviousDays = 50

        jsonFile = open("model%s.json"%ticker, "r")
        loadedModelJson = jsonFile.read()
        jsonFile.close()
        model = model_from_json(loadedModelJson)
        model.load_weights("model%s.h5"%ticker)
        print("Loaded model")

        #Sliding window algorithm used to predict the close values for the next 30 days. Will use 

        #Gets the scaled close values for the last ... days in dfScaled as an np array with one row (and as many columns as necessary)
        previousDays = dfScaled[len(dfScaled) - numberOfPreviousDays:].reshape(1, -1)

        #Converts this to a list
        previousDaysTemp = list(previousDays)
        previousDaysTemp = previousDaysTemp[0].tolist()

        #List to store the future predictions
        futurePredictions = []

        #Predicting next 30 days
        daysToPredict = 30

        for i in range(daysToPredict):
            if(len(previousDaysTemp) > numberOfPreviousDays):
                previousDays = np.array(previousDaysTemp[1:])
                previousDays = previousDays.reshape((1, numberOfPreviousDays, 1))
                y = model.predict(previousDays, verbose = 0)
                previousDaysTemp.extend(y[0].tolist())
                previousDaysTemp = previousDaysTemp[1:]
                futurePredictions.extend(y.tolist())
            else:
                previousDays = previousDays.reshape((1, numberOfPreviousDays, 1))
                y = model.predict(previousDays, verbose = 0)
                previousDaysTemp.extend(y[0].tolist())
                futurePredictions.extend(y.tolist())

        dfScaledAndFuturePredictions = dfScaled.tolist()
        dfScaledAndFuturePredictions.extend(futurePredictions)
        dfAndFuturePredictions = scaler.inverse_transform(dfScaledAndFuturePredictions).tolist()
        last100dfPred = dfAndFuturePredictions[len(dfAndFuturePredictions) - 100:]
        last100dfPredList = [item[0] for item in last100dfPred]
        last70df = last100dfPredList[:70]
        predictions = last100dfPredList[70:]
    else:
        last70df = []
        predictions = []
    # Convert the 100 days data to a list
    last100Days = np.arange(-69, 31)
    last100DaysList = last100Days.tolist()
    return last70df, predictions, last100DaysList


# Cache the data for 5 minutes so the page loads quicker on subsequent visits
@cache.memoize(300)
def company_chart(ticker):
    # Gets AlphaVantage's overview of the company from their ticker (includes name, description, etc.)
    overview = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker.replace('.', '-')}&apikey=QCLLY08TISZBMXL9").json()
    name = overview["Name"]
    description = overview["Description"]
    industry = overview["Sector"].title()
    # Query AlphaVantage for the daily price data of the company (price at open, close, etc.)
    daily_price_data = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=QCLLY08TISZBMXL9").json()["Time Series (Daily)"]
    daily_prices = []
    days = []
    # Extract the price at close data from the JSON dictionary returned by the above query
    for day in daily_price_data:
        daily_prices.insert(0, float(daily_price_data[day]["4. close"]))
        days.insert(0, day)
    # Query AlphaVantage for the weekly price data of the company (price at open, close, etc.)
    weekly_price_data = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={ticker}&apikey=QCLLY08TISZBMXL9").json()["Weekly Time Series"]
    weekly_prices = []
    weeks = []
    # Extract the price at close data from the JSON dictionary returned by the above query
    if len(weekly_price_data) > 100:
        # Limits the amount of data if there are more than 100 weeks available
        weekly_price_data= dict(islice(weekly_price_data.items(),100))
    for week in weekly_price_data:
        weekly_prices.insert(0, float(weekly_price_data[week]["4. close"]))
        weeks.insert(0, week)
    # Query AlphaVantage for the monthly price data of the company (price at open, close, etc.)
    monthly_price_data = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey=QCLLY08TISZBMXL9").json()["Monthly Time Series"]
    monthly_prices = []
    months = []
    # Extract the price at close data from the JSON dictionary returned by the above query
    if len(monthly_price_data) > 100:
        # Limits the amount of data if there are more than 100 months available
        monthly_price_data = dict(islice(monthly_price_data.items(),100))
    for month in monthly_price_data:
        monthly_prices.insert(0, float(monthly_price_data[month]["4. close"]))
        months.insert(0, month)

    # Calculate daily returns - used to calculate volatility
    daily_returns = [0] + [((p2 / p1) - 1) for p1, p2 in zip(daily_prices, daily_prices[1:])]

    # Calculate volatility as the standard deviation of daily returns (as percent)
    volatility = np.std(daily_returns) * 100

    # Finds the earliest available day for which AV has data on that company and formats it so that it can be passed to the API call to get news sentiment
    earliest_day = days[0].replace("-", "")

    # Get the daily sentiment data for the company from AlphaVantage, passing in earliest_day as the time it should check from
    daily_sentiment_data = requests.get(f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={earliest_day}T0000&apikey=QCLLY08TISZBMXL9").json()["feed"]
    daily_sentiment = []
    days2 = []
    # Stores the sentiment and date from each of the returned articles
    for article in daily_sentiment_data:
        daily_sentiment.insert(0, float(article["overall_sentiment_score"]))
        datetime_obj = datetime.strptime(article["time_published"], "%Y%m%dT%H%M%S")
        days2.insert(0, datetime_obj.strftime("%Y-%m-%d %H:%M"))
    return daily_prices, daily_sentiment, days, days2, volatility, name, description, industry, weekly_prices, weeks, monthly_prices, months, daily_sentiment_data

# Logic for the settings page
@app.route('/settings')
def settings():
    following = []
    # Checks if the user is logged in
    if not hasattr(current_user, "email"):
        return redirect("/login")
    # Logic for hiding the email in the settings page
    # Replaces the middle characters of the name anad domain of the email with *
    email_split = current_user.email.split("@")
    domain_split = email_split[1].split(".")
    hidden_email = email_split[0][0] + '*'*(len(email_split[0])-2) + email_split[0][len(email_split[0])-1] + '@' + domain_split[0][0] + '*'*(len(domain_split[0])-2) + domain_split[0][len(domain_split[0])-1]
    for i in range(len(domain_split)-1):
        hidden_email += '.' + domain_split[i+1]
    # Gets the companies followed by the user. Used for the following dropdown in the menu bar
    compQuery = get_followed_companies()
    for comp in compQuery:
        following.append([comp.name, comp.ticker])
    return render_template("settings.html", hidden_email=hidden_email, following=following)

# Logic for changing email
@app.route('/settings/e', methods=["GET", "POST"])
def change_email():
    # Checks if the user is not logged in (since you can navigate to the page by typing in the address without being logged in)
    if not hasattr(current_user, "email"):
        return redirect("/login")
    # Get the values from the forms on the page
    old_password = request.form.get("password-curr")
    email = request.form.get("email")
    if email == current_user.email:
        flash("Cannot update email: you must enter a different email", 'error')
    else:
        # Get the current password, check if it matches the provided one
        user = Users.query.filter_by(email=current_user.email).first()
        if user and security.check_password_hash(user.password_hash, old_password):
            # If it does match then update the email stored in the db
            user.email = email
            db.session.commit()
            flash("Email updated!", "success")
        else:
            flash("Password incorrect", "error")
    return redirect('/settings')

# Logic for changing password
@app.route('/settings/p', methods=["GET", "POST"])
def change_password():
    # Checks if the user is not logged in (since you can navigate to the page by typing in the address without being logged in)
    if not hasattr(current_user, "email"):
        return redirect("/login")
    old_password = request.form.get("password-curr")
    new_password = request.form.get("password-new")
    new_password_hash = security.generate_password_hash(new_password)
    conf_password = request.form.get("password-conf")
    user = Users.query.filter_by(email=current_user.email).first()
    # Checks if the new password is new
    if security.check_password_hash(user.password_hash, new_password):
        flash("New password should be different to current password", "error")
        return redirect('/settings')
    # Checks if the new and confirmed passwords are the same
    if not security.check_password_hash(new_password_hash, conf_password):
        flash("Passwords do not match", "error")
        return redirect('/settings')
    # If none of those issues occurred, and the current password matches the one stored, then update the one stored in the db
    if user and security.check_password_hash(user.password_hash, old_password):
        user.password_hash = new_password_hash
        db.session.commit()
        flash("Password updated!", "success")
    else:
        flash("Password incorrect", "error")
    return redirect('/settings')

# Logic for changing name stored in db
@app.route('/settings/n', methods=["GET", "POST"])
def change_name():
    # Checks if the user is not logged in (since you can navigate to the page by typing in the address without being logged in)
    if not hasattr(current_user, "email"):
        return redirect("/login")
    name = request.form.get("name")
    user = Users.query.filter_by(email=current_user.email).first()
    # Update the name stored in the db if the user tuple is found
    if user is not None:
        user.name = name
        db.session.commit()
        flash("Name updated!", "success")
    else:
        flash("Error updating name", "error")
    return redirect('/settings')

# Logic for deleting a user's account
@app.route('/settings/d', methods=["GET", "POST"])
def delete_account():
    # Checks if the user is not logged in (since you can navigate to the page by typing in the address without being logged in)
    if not hasattr(current_user, "email"):
        return redirect("/login")
    old_password = request.form.get("password-curr")
    conf_password = request.form.get("password-conf")
    user = Users.query.filter_by(email=current_user.email).first()
    # Checks the confirmation password matches
    if old_password != conf_password:
        flash("Passwords do not match", "error")
        return redirect('/settings')
    # Checks the submitted password is correct
    if user and security.check_password_hash(user.password_hash, old_password):
        followQuery = UserFollow.query.filter_by(userid=user.id).all()
        lastQuery = LastUpdated.query.filter_by(userid=user.id).first()
        # Delete all insances of this user account from the UserFollow table
        for row in followQuery:
            db.session.delete(row)
        # Log the user out
        logout_user()
        # Delete the tuple in LastUpdated with the same user id
        if lastQuery is not None:
            db.session.delete(lastQuery)
        # Delete the user tuple from the Users table
        db.session.delete(user)
        db.session.commit()
        flash("Account Deleted", "success")
        return redirect('/')
    else:
        flash("Password incorrect", "error")
        return redirect('/settings')
    
# Logic for following a company
@app.route('/company/<ticker>/#<follow>', methods=["GET", "POST"])
def follow_company(ticker, follow):
    # Checks the user is logged in
    if not hasattr(current_user, "email"):
        flash("Please log in")
        return redirect("/company/"+str(ticker))
    usrQuery = Users.query.filter_by(email=current_user.email).first()
    compQuery = Company.query.filter_by(ticker=ticker).first()
    # Check the company and user account exist
    if compQuery is not None and usrQuery is not None:
        # Logic to follow a company
        if follow == "Follow":
            # Adds the user and company ids to the UserFollow table
            db.session.add(UserFollow(usrQuery.id,compQuery.id))
            db.session.commit()
            # Deletes the memo for the home page, so the data must be recalculated (since there are new companies to display)
            cache.delete_memoized(getNewsFeed)
            flash("Followed Company!", "success")
        else:
            # Deletes the entry from the UserFollow table corresponding to the user following this company
            toDelete = UserFollow.query.filter(UserFollow.userid==usrQuery.id, UserFollow.companyid==compQuery.id).first()
            if toDelete is not None:
                db.session.delete(toDelete)
                db.session.commit()
                # Deletes the memo for the home page, so the data must be recalculated (since there are fewer companies to display)
                cache.delete_memoized(getNewsFeed)
                flash("Unfollowed Company!", "success")
            else:
                flash("Error in Following Company", "error")
    else:
        flash("Error in Following Company", "error")
    return redirect("/company/"+str(ticker))

# Function to upload a profile picture
@app.route('/upload', methods=['POST'])
def upload_file():
    # Checks a file was properly submitted
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/settings')

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect('/settings')

    if file:
        # Secure_filename returns a sanitised filename (removes special characters)
        filename = secure_filename(file.filename)
        # Save the file to the static folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Set the user's associated profile picture to the one that has just been uploaded
        user = Users.query.filter_by(email=current_user.email).first()
        if user:
            user.pic = filename
            db.session.commit()
            flash('File uploaded successfully')
        else:
            flash("Error in uploading file","error")
        return redirect('/settings')

# Occurs when the user presses the submit button on the search bar
@app.route("/searchsubmit", methods=["GET", "POST"])
def searchsubmit():
    query = request.form.get("search")
    # If the user does not submit anything then the query gets replaced with the wildcard character (%)
    if query == "":
        return redirect(f"/search/%")
    return redirect(f"/search/{query}")

# Handles searching
@app.route("/search/<query>", methods=["GET" ,"POST"])
def search(query):
    following = []
    # Query the company table for any company name/ticker which is similar to the query
    search_results = Company.query.filter(Company.name.like(f"%{query}%") | Company.ticker.like(f"%{query}%")).all()
    # Get the companies followed by the user, used for the following dropdown in the menu bar
    compQuery = get_followed_companies()
    print(search_results)
    for comp in compQuery:
        following.append([comp.name, comp.ticker])
    return render_template("search.html", search_results=search_results, query=query, following=following)

# Gets the company tuples of the companies followed by the user
def get_followed_companies():
    usrQuery = Users.query.filter_by(email=current_user.email).first()
    followQuery = UserFollow.query.filter_by(userid=usrQuery.id).all()
    queries = []
    for result in followQuery:
        queries.append(Company.query.filter_by(id=result.companyid).first())
    return queries

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)