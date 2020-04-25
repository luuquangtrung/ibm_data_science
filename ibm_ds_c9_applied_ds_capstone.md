# COURSE 9: APPLIED DATA SCIENCE CAPSTONE

## Syllabus:
	Week 1 - Introduction to Capstone Project
		Introduction to Capstone Project
		Location Data Providers
		Signing-up for a Watson Studio Account
		Peer-review Assignment: Capstone Project Notebook
	Week 2 - Foursquare API
		Introduction to Foursquare
		Getting Foursquare API Credentials
		Using Foursquare API
		Lab: Foursquare API
		Quiz: Foursquare API
	Week 3 - Neighborhood Segmentation and Clustering
		Clustering
		Lab: Clustering
		Lab: Segmenting and Clustering Neighborhoods in New York City
		Peer-review Assignment: Segmenting and Clustering Neighborhoods in Toronto
	Week 4 - Capstone Project
	Week 5 - Capstone Project (Cont'd)


## Week 1 - Introduction to Capstone Project
### 1.1. Introduction to Capstone Project
### 1.2. Location Data Providers
### 1.3. Signing-up for a Watson Studio Account
### 1.4. Peer-review Assignment: Capstone Project Notebook


Week 2 - Foursquare API

* Website: foursquare.com
* Personal account:
	* 105M+ points of interests
	* 1 photos & 1 tips per venue
	* 2 queries per second (QPS)
	* 1 app per account
	* API call quota: 99,500 regular calls + 500 premium calls per day

* RESTful API
	* Communicate with the database via groups and endpoints in the form of a URI (Uniform Resource Identifier)
	* Any call request is composed of: 

```json
	https://api.foursquare.com/v2/tips/
								 /users
								 /venues
```
	* Every time a call request is made, you have to pass your developer account credentials, which are
		* Client ID 
		* Client Secret 
		* Version of the API, which is simply a date

**Example 1**: You are at the Conrad Hotel, NY, and you wanna go for a cup of coffee

```json
	// Make a regular call 
	https://api.foursquare.com/v2/venues/search?
		client_id=12345
		&client_secret=****
		&v=20180602
		&ll=40.73,-74.01&query=coffee
```
* Result: We're gonna get a JSON file of the venues that match our query
* With personal account: we can make 99,500 regular calls per day
* Result of each venue: 
	* Name
	* Unique ID
	* Location
	* Category

**Example 2**: You've decided to go to Kaffe 1668, but you wanna learn more about the coffee shop

```json
	// Premium call 
	// Now add the unique id of Kaffe 1668 to the call request
	// unique id = 49ccd495dfsdfsdf4e
	https://api.foursquare.com/v2/venues/49ccd495dfsdfsdf4e?
		client_id=12345
		&client_secret=****
		&v=20180602
		&ll=40.73,-74.01&query=coffee
```
* JSON result of the venue "Kaffe 1668" contains: 
	* Name
	* URL or website 
	* Unique ID
	* Average rating
	* Location
	* Contact information
	* Menu
	* Statistics in terms of the number of check-ins 
	* Tips posted about the venue

* Permium call: 2 tips and photos per venue
* With personal account: we can make 500 premium calls per day



* Summary: Using Foursquare API we can
	1. Search for a specific type of venue around a given location (Regular)
	2. Learn more about a specific venue (Premium)
	3. Learn more about a specific Foursquare user (Regular)
	4. Explore a given location (Regular)
	5. Explore trending venues around a given location (Regular)


Foursquare has more than one endpoint. There is the explore endpoint to explore a given location and there is the search endpoint which is used to search for particular venues around a given location

### Lab: Foursquare API
### Quiz: Foursquare API




1. The explore endpoint is used to search for venues in the vicinity of a particular location.
	True.
	*False.* (The search endpoint is the appropriate endpoint to use)

2. Which of the following parameters need to be passed every time you make a call to the Foursquare API?
	*Your client secret.*
	*Your client ID.*
	*The version of the API.*
	Search query.
	The latitude and longitude values of a location.

3. Using a personal (free) developer account, you can make 500 premium calls per hour.
	True.
	*False.* (per day)

4. Using a personal (free) developer account, you can access 1 tip and 1 photo per venue.
	*True.*
	False.

5. You can access a venue tips and photos through regular calls.
	True.
	*False.*

What endpoint is used to explore what is near a particular location?
	You don't use any function. You can simply pass the unique ID of the location. (false)
	The find endpoint.
	The category endpoint.
	The search endpoint.
	The explore endpoint.



Which of the following (information, call type) pairs are correct?
	Venue tips, Regular (Premium)
	*Venue details, Premium* 
	*Venue menu, Premium*
	Venues nearby, Premium (Exploring a location is done using a regular call)
	*Venue tips, Premiun*




Week 3 - Neighborhood Segmentation and Clustering
### 3.1. Clustering
### Lab: Clustering
### Lab: Segmenting and Clustering Neighborhoods in New York City
### Peer-review Assignment: Segmenting and Clustering Neighborhoods in Toronto

## Week 4 - Capstone Project
## Week 5 - Capstone Project (Cont'd)