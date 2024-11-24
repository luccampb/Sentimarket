from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# Users table
class Users(UserMixin, db.Model):
    __tablename__='Users'
    # Defining the user attributes
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.Text())
    name = db.Column(db.Text())
    pic = db.Column(db.Text())
    
    # Required for flask_sqlalchemy
    def __init__(self, email, password_hash, name, pic):  
        self.email=email
        self.password_hash=password_hash
        self.name=name
        self.pic=pic

# Table which records the companies that users follow
class UserFollow(db.Model):
    __tablename__='UserFollow'
    # Creates a composite key for the table, comprising of the userid and companyid fields
    __table_args__ = (
        db.PrimaryKeyConstraint("userid", "companyid"),
    )
    # Defining the UserFollow attributes
    userid = db.Column(db.Integer, db.ForeignKey("Users.id"), nullable=False)
    companyid = db.Column(db.Integer, db.ForeignKey("Company.id"), nullable=False)

    # Required for flask_sqlalchemy
    def __init__(self, userid, companyid):  
        self.userid=userid
        self.companyid=companyid

# Table which stores the companies
class Company(db.Model):
    __tablename__='Company'
    # Defining the Company attributes
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    industry = db.Column(db.String(50), nullable=False)

    # Required for flask_sqlalchemy
    def __init__(self, ticker, name, industry):  
        self.ticker=ticker
        self.name=name
        self.industry=industry

# Stores the last time a user received a notification for a company which they follow (used to check if a news story is new)
class LastUpdated(db.Model):
    __tablename__="LastUpdated"
    # Defining the LastUpdated attributes
    userid = db.Column(db.Integer, db.ForeignKey("Users.id"), nullable=False, primary_key=True)
    timestamp = db.Column(db.Integer)

    # Required for flask_sqlalchemy
    def __init__(self, userid, timestamp):
        self.userid = userid
        self.timestamp = timestamp