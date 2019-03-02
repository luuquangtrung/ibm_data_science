# COURSE 5: DATABASES AND SQL FOR DATA SCIENCE

**NOTE**: It's faster and better using [w3school.com](https://www.w3schools.com/sql/trysql.asp?filename=trysql_op_in) to test SQL queries than IBM Db2


## WEEK 1: Introduction to Databases and Basic SQL

Summary:
Statement | Category
----------|----------------------------------
`CREATE`  | *DDL*
`INSERT`  | DML
`SELECT`  | DML
`UPDATE`  | DML
`DELETE`  | DML

* `DDL`: Data Definition Languague
* `DML`: Data Manipulation Languague

Video 1: Basic SQL (5 commands) and Relational Databases

* SQL: Structured (English) Query Languague
	* A languague used for relational databases to query data

* Databases:
	* A repository of data
	* Provides the functionality for adding, modifying, and querying that data
	* Different kinds of databases store data in *different* forms
	* Data stored in tabular form is a *relational* database
	* DBMS (Database Management System): A set of software tools for the data in the databases
	* Terms: DB, DB server, and DBMS are often used interchangeably

* Quiz:
	* A database is a repository or logically coherent collection of data with some inherent meaning
	* Typically comes with functionality for adding, modifying and querying data
	* SQL or Structured Query Language is commonly used for accessing data in relational databases	

* RDBMS (Relational DBMS):
	* A set of software tools to control data: access, organization, storage
	* Example: `Db2 Warehouse on Cloud` or `DB2 Express-C`
	* *Note: A relational database stores data in a tabular format, i.e., in rows and columns. But not all types of databases use the tabular format*

* Basic 5 SQL commands: `CREATE`, `INSERT`, `SELECT`, `UPDATE`, `DELETE`

Video 2: How to create a database instance?

* Cloud database:
	* Advantages: Ease of use, Scalability, Disaster Recovery

* DB service instance:
	* DB services are logical abastractions for managing workloads in a DB
	* An instance of the Cloud DB operates as a service that handles all apps requests to work with the data in any of the DBs managed my that instance
	* The `Db2 Warehouse DB` (IBM) can be provisioned on IBM Cloud and for AWS
	* Connection configuration resources:	
		* Host name: `dashdb-entry-yp-dal10-01.services.dal.bluemix.net`
		* Port number: 50000
		* Database name: BLUDB
		* User ID: dash7000
		* Password
		* Version: Compatible with DB2 for Linux, UNIX, Windows. v11.1 or later

Video 3: CREATE Table Statement
Outline: 
	* Distinguish between: 
		1. Data Definition Languague (DDL) statements: Define, Change, or Drop data
		-> *CREATE is DDL*
		2. Data Manipulation Languague (DML) statements: Read, Modify data
	* Explain how the Entity name and Attributes are used to create a relational DB table


* Constraints in table:
	* The Primary Key of a relational table uniquely identifies each row in a table
	* First and Last name must be not null


Example: General syntax:
```sql
create table TABLENAME (
    COLUMN1 datatype,
    COLUMN2 datatype,
    COLUMN3, datatype,
        ...
    ) ;
```

```sql

-- If the table you are trying to create already exists in the database you will get 
-- an error indicating "table XXX.YYY already exists". To circumvent this error, either create a table with a different name or first DROP the existing table
-- WARNING: before dropping a table ensure that it doesn't contain important data that can't be recovered easily.
drop table COUNTRY;						

create table COUNTRY (
    ID integer PRIMARY KEY NOT NULL, 	-- It cannot contain a NULL or empty value
    CCODE char(2),						-- Country code of type char (2 letters)
    NAME varchar(60)					-- A variable length country name column
    );
```


Video 4: SELECT Statement
Outline:
	* Retrieve data from a relational DB table
	* Define the use of a predicate
	* Identify syntax of `SELECT` using `WHERE` clause
	* List the comparision operators supported by a RDRMS

1. Retrieve data from table:

Example of a Relational DB Table:

Table `book_table`:
---------------------------------------------
book_id	| title
---------------------------------------------
B1		| Getting started with DB2 Express-C
B2		| Database fundamentals
B3		| Harry Potter
---------------------------------------------

```sql
select * from <tablename>
db2 => select * from book_table
```

`SELECT` statement:
	* `SELECT` is a *DML* query used to read and modify data
	* Output of SELECT is called the `Result Set` or a `Result Table`

We can retrieve just a subset of columns:
The order of the columns in the query will be the display order 

```sql
select <column1>, <column2> from <tablename>
db2 => select book_id, title from book_table
```

`WHERE` clause:
	* Restricting the Result Set
	* Always requires a Predicate: 
		* Evaluates to: True, False, or Unknown,
		* Used in the search condition of the WHERE clause

```sql
db2 => select book_id, title from book_table where book_id = 'B1'
-- Comparators: =, >, <, >=, <=, <>
```


USEFUL OPTIONS USED WITH `SELECT` STATEMENT:

* `COUNT()`: to get the number of rows that match the query criteria
```sql 
select COUNT(*) from tablename
-- Get nb of rows where the medal recipient is from CANADA:
select COUNT(COUNTRY) from MEDALS where COUNTRY='CANADA' 
```

* `DISTINCT`: to remove duplicate values from a result set
```sql 
-- Retrieve the list of unique countries that received GOLD medals:
select DISTINCT COUNTRY from MEDALS where MEDALTYPE = 'GOLD'
```

* `LIMIT`: to restrict the number of rows retrieved from the database
```sql 
-- Get only 5 rows of the result
select * from MEDALS where YEAR = 2018 LIMIT 5
```


Video 5: INSERT Statement
Outline:
	* Syntax of `INSERT`
	* Explain 2 methods to add rows to a table

`INSERT` statement:
	* `INSERT` is a *DML* query used to read and modify data

Table `book_table`:
---------------------------------------------
book_id	| title
---------------------------------------------
B1		| Getting started with DB2 Express-C
B2		| Database fundamentals
B3		| Harry Potter
---------------------------------------------

```sql 
-- Syntax:
insert into [tablename] <([columnname1],...)> values ([value1],...)
-- Example:
insert into book_table 
	(book_id, title)				-- Column names
	values
	('B4', 'The Lord of the Rings') -- Nb of values must match the nb of column name
```

We can also add multiple rows at once:
```sql 
insert into book_table 
	(book_id, title)				
	values
	('B4', 'The Lord of the Rings') 
	('B5', 'The Hobbit') 
```

Video 6: UPDATE and DELETE Statements
Outline:
	* Syntax of `UPDATE` and `DELETE`
	* Importance of `WHERE` clause in both 2 statements

Principles:
	* `UPDATE` and `DELETE` are both *DML* query used to read and modify data


Table `book_table`:
---------------------------------------------
book_id	| title 					| author			
---------------------------------------------
B1		| Getting started with DB 	| Peter
B2		| Database fundamentals	    | Kane
B3		| Harry Potter 				| Jane
---------------------------------------------

```sql 
-- Syntax:

-- update values in rows
update [tablename] set [[columnname1] = [value1]] <where [condition]> 

-- delete rows	
delete from [tablename] <where [condition]>  

-- Example:
update book_table 
	set title = "Harry Potter and the Philosopher's Stone"
		author = "J. K. Rowling"
	where book_id = B3


delete from book_table 
	where book_id in ('B1', 'B3') -- Will delete row B1 and B3
```

*NOTE*: Be careful, if `WHERE` clause is not specified
	* With `UPDATE`: ALL the rows will be updated!
	* With `DELETE`: ALL the rows will be removed!



## WEEK 2: Advanced SQL

### Video 1: Using String Patterns, Ranges

Table `book_table`:
-----------------------------------------------------
book_id	| title 					| author | pages			
-----------------------------------------------------
B1		| Getting started with DB 	| Peter  | 200
B2		| Database fundamentals	    | Kane   | 295
B3		| Harry Potter 				| Jane   | 300
-----------------------------------------------------

* What if we can't remember the name of the author, just only the first name starts with "R"? In a relational DB, we can use string patterns to search data rows that match this condition

```sql 
-- EXAMPLE 1:
db2 => select firstname from author where firstname like 'R%'
-- The percent sign % to define missing letters, and can be placed
-- before, after, or both before and after the pattern (i.e., 'R')

-- EXAMPLE 2:
db2 => select title, pages from book_table where pages >= 290 AND pages <= 300
-- Instead of the above, we can rather use the following: More easier and quicker
db2 => select title, pages from book_table where pages between 290 and 300

-- EXAMPLE 3:
db2 => select firstname, lastname, country from author where 
		country='AU' OR country='BR'
-- Use this instead:
db2 => select firstname, lastname, country from author where 
		country IN ('AU','BR')
```


### Video 2: Sorting Result Sets
* How to sort result sets by either ascending or descending order
* How to indicate which column to use for the sorting order

**Summary:**
* By default `ORDER BY` sorts the data in ascending order.
* We can use the keyword `DESC` to sort the data in descending order and the keyword `ASC` to sort in ascending order.

```sql 
db2 => select title from book_table order by title 	  --by default: ascending order (ABC)
db2 => select title, pages from book_table order by 2 --id of the column to be sorted
```


### Video 3: Grouping Result Sets
* How to eliminate duplicates from a result set
* How to restrict a result set: Using `DISTINCT`, `GROUP BY`, and `HAVING` clause

1. `DISTINCT` clause
```sql 
-- get all countries including duplicates
db2 => select country from author order by 1

-- only ger unique country
db2 => select distinct(country) from author  
```

2. `GROUP BY` clause
	* `HAVING` clause works only with the `GROUP BY` clause

```sql 
-- get all countries including duplicates
db2 => select country from author order by 1 

/* 
	COUNTRY
	--------
	AU
	BR
	...
	CN 
	CN 
	...
*/

-- count nb of authors for each country
db2 => select country, count(country) from author 
	group by country  

/* 
	COUNTRY  	2
	-------- 	--------
	AU 			1
	BR 			5
	CN 			6
*/

-- count nb of authors for each country and name the column
db2 => select country, count(country) as count from author 
	group by country 

/* 
	COUNTRY  	COUNT
	-------- 	--------
	AU 			1
	BR 			5
	CN 			6
*/

-- furthermore: restrict the country having more than 4 authors
db2 => select country, count(country) as count from author 
	group by country having count(country) > 4 

/* 
	COUNTRY  	COUNT
	-------- 	--------
	BR 			5
	CN 			6
*/
```


### READING: Built-in Functions, Dates, Timestamps

* While it is very much possible to first fetch data from a DB and then perform operations on it, most databases come with built-in functions to allow you to perform operations on data right within the database.
* Using DB functions can significantly reduce the amount of data that needs to be retrieved from the database (reduces network traffic and use of bandwidth) and may also be faster. (It is also possible to create your own functions in the database but that is beyond the scope of this course).

**Example**: Table `PET_SALE`:
-----------------------------------------------------------------
ID		| ANIMAL 		| QUANTITY	| SALEPRICE		| SALEDATE		
integer	| varchar(20)	| integer	| decimal(6,2)	| date
-----------------------------------------------------------------
1		| Cat  			| 9 		| 450.09		| 2018-05-29
2		| Dog    		| 3   		| 666.66		| 2018-05-20
3		| Parrot 		| 1   		| 300.00		| 2018-06-04
-----------------------------------------------------------------

#### Aggregate or Column Functions
* An aggregate function takes a collection of like values (such as all of the values in a column) as input and returns a single value (or `NULL`)
* Examples of aggregate functions include `SUM()`, `MIN()`, `MAX()`, `AVG()`, etc.


```sql
-- 1. Add up all the values in the SALEPRICE column:
select SUM(SALEPRICE) from PETSALE

-- 2. Now explicitly name the output column SUM_OF_SALEPRICE :
select SUM(SALEPRICE) as SUM_OF_SALEPRICE from PETSALE

-- 3. Maximum QUANTITY of any ANIMAL sold:
select MAX(QUANTITY) from PETSALE

-- 4. Average value of SALEPRICE :
select AVG(SALEPRICE) from PETSALE

-- 5. Average SALEPRICE per 'Dog':
select AVG(SALEPRICE/QUANTITY) from PETSALE where ANIMAL = 'Dog'
```

* Notice above that we can perform mathematical operations between columns. In this case the `SALEPRICE` is for multiple units so we first divide the `SALEPRICE` by the `QUANTITY` of the sale.


#### SCALAR and STRING FUNCTIONS

Scalar functions perform operation on individual values.

```sql
-- 6. Round UP/DOWN every value in SALEPRICE column to nearest integer:
select ROUND(SALEPRICE) from PETSALE

/*
There is a class of Scalar functions that can be used 
for operations on string (CHAR and VARCHAR) values:
*/

-- 7. Retrieve the length of each value in ANIMAL column:
select LENGTH(ANIMAL) from PETSALE

-- 8. Retrieve the ANIMAL column values in UPPERCASE format:
select UCASE(ANIMAL) from PETSALE

-- 9. Use the function in a WHERE clause:
select * from PETSALE where LCASE(ANIMAL) = 'cat'

/*
The above is useful if you are not sure whether the 
values are stored in upper, lower or mixed case.
*/

-- 10. Use DISTINCT() function to get unique values:
select DISTINCT(UCASE(ANIMAL)) from PETSALE
```

#### (OPTIONAL) Date, Time functions
* Most databases contain special datatypes for dates and times
* `Db2` contains `DATE`, `TIME`, and `TIMESTAMP` types:
	* `DATE` has 8 digits: `YYYYMMDD`
	* `TIME` has six digits: `HHMMSS`
	* `TIMESTAMP` has 20 digits: `YYYYXXDDHHMMSSZZZZZZ` where `XX` represents month and `ZZZZZZ` represents microseconds
* Functions exist to extract the `DAY`, `MONTH`, `DAYOFMONTH`, `DAYOFWEEK`, `DAYOFYEAR`, `WEEK`, `HOUR`, `MINUTE`, `SECOND`

```sql
-- 11. Extract the DAY portion from a date:
select DAY(SALEDATE) from PETSALE where ANIMAL = 'Cat'

-- 12. Get the number of sales during the month of may (i.e. month 5):
select COUNT(*) from PETSALE where MONTH(SALEDATE)='05'

-- You can also perform DATE or TIME arithmetic.

-- 13. What date is it 3 days after each saledate [maybe you want to 
-- know this because the order needs to be processed with 3 days]:
select (SALEDATE + 3 DAYS) from PETSALE

-- Special registers CURRENT TIME and CURRENT DATE are also available:

-- 14. Find how many days have passed since each SALEDATE till now:
select (CURRENT DATE - SALEDATE) from PETSALE
```

### READING: Sub-queries and Nested Selects

* Sub-queries or sub-selects are like regular queries but placed within paranthesis and nested inside another query. This allows you to form more powerful queries than would have been otherwise possible
* Consider the `EMPLOYEES` table from the previous lesson. Lets say we want to retrieve the list of employees who earn more than the average salary. To do so we could try:

```sql
select * from employees where salary > avg(salary)
```

However running this query will result in an error like:
```sql
SQL0120N   Invalid use of an aggregate function or OLAP function.SQLCODE=-120, SQLSTATE=42903
```

* One of the limitations of built-in aggregate functions like `AVG()` is that they cannot always be evaluated in the `WHERE` clause.
* So to evaluate the `AVG()` function in the where clause we can make use of a sub-Select expression like the follows:

```sql
select EMP_ID, F_NAME, L_NAME, SALARY from employees 
	where SALARY < (select AVG(SALARY) from employees);
```

**Example**: Table `employees`:
----------------------------------------------------
EMP_ID		| F_NAME 		| L_NAME	| SALARY		
----------------------------------------------------
E1008		| Bharath  		| Gupta		| 65000.00		
E1007		| Mary			| Thomas   	| 65000.00
E1004		| Santosh 		| Kumar   	| 65000.00
----------------------------------------------------

The `IN` operator can also be used and there can be multiple leves of sub-queries, such as:

```sql
select EMP_ID, F_NAME, L_NAME, DEP_ID from employees 
 where DEP_ID IN  
 ( select DEP_ID from employees where DEP_ID > 
         ( select MIN(DEP_ID) from employees ) 
 );
```

The sub-select doesn't just have to go in the where clause, it can also go in other parts of the query such as in the list of columns to be selected:

```sql
select EMP_ID, SALARY,
  ( select AVG(SALARY) from employees ) 
  AS AVG_SALARY
  from employees ;
```

Another option is to make the sub-query be part of the `FROM` clause:

```sql
select * from 
 ( select EMP_ID, F_NAME, L_NAME, DEP_ID 
 from employees) ;
```

### READING: Working with Multiple Tables

* There are several ways to access multiple tables in the same query:
	1. Sub-queries
	2. Implicit `JOIN`
	3. `JOIN` operators (`INNER JOIN`, `OUTER JOIN`, etc.)
In this lesson we will examine the first two options. The third is covered in more details in additional lessons.

1. Sub-queries
In a previous lesson we learned how to use sub-queries. Now let's use sub-queries to work with multiple tables. Lets consider the `EMPLOYEES` and department tables from the previous lesson.

**Example**: Table `EMPLOYEES`:
---------------------------------------------------------------------------------------------------
EMP_ID | F_NAME | L_NAME | SSN | B_DATE 	| SEX | ADDRESS | JOB_ID | SALARY | MANAGER_ID | DEP_ID		
---------------------------------------------------------------------------------------------------
E1008  | Bhara  | Gupta  | 123 | 1972-07-31 | F   | 1 av... | 200    | 65000  | 30002	   | 5	
E1007  | Mary 	| Thomas | 123 | 1972-07-31 | F   |	1 av... | 400    | 65000  | 30002      | 5
E1004  | San  	| Kumar  | 123 | 1972-07-31 | M   | 1 av... | 500    | 65000  | 30002 	   | 7
---------------------------------------------------------------------------------------------------


```sql
SELECT F_NAME FROM EMPLOYEES 
	WHERE SALARY = ( SELECT MAX(SALARY) FROM EMPLOYEES )
```


**Example**: Table `DEPARTMENTS`:
---------------------------------------------
DEPT_ID_DEP | DEP_NAME | MANAGER_ID | LOC_ID		
---------------------------------------------
5			| Software | 30002		| L0002
7			| Design   | 30003		| L0003
---------------------------------------------


```sql
SELECT * FROM EMPLOYEES 
	WHERE DEP_ID = ( SELECT MAX(DEPT_ID_DEP) FROM DEPARTMENTS )
```


If we want to retrieve only the employee records that correspond to departments in the DEPARTMENTS table, we can use:

```sql
select * from employees 
  where DEP_ID IN
  ( select DEPT_ID_DEP from departments );
```

Let's say we want to retrieve only the list of employees from a specific location. We do not have any location information in the EMPLOYEES table but the DEPARTMENTS table has a column called LOC_ID. So we can use a sub-query from the DEPARTMENTS table as input to the EMPLOYEE table query:

```sql
select * from employees 
  where DEP_ID IN
   ( select DEPT_ID_DEP from departments 
     where LOC_ID = 'L0002' );
```

Now let's retrieve the department ID and name for Empolyees who earn more than 70000:

```sql
select DEPT_ID_DEP, DEP_NAME from departments
 where DEPT_ID_DEP IN
  ( select DEP_ID from employees 
    where SALARY > 70000 ) ;
```

2. Implicit join

Here we specify two tables in the query (but note we are not explicitly using the `JOIN` operator). The result is a full join (or cartesian join), because every row in the first table is joined with every row in the second table. If you examine the result set you will see more rows than in both tables individually.

```sql
select * from employees, departments;
```

We can use additional operands to limit the result set. In the following example we limit the result set to only rows with matching department IDs:

```sql
select * from employees, departments 
  where employees.DEP_ID = departments.DEPT_ID_DEP;
```

Notice that in the where clause we pre-fixed the name of the column with the name of the table to fully qualify the column name since it's possible that the different tables could have the some column names that are exactly the same. Since the table names can be sometimes long, we can use shorther aliases for table names as follows:

```sql
select * from employees E, departments D 
  where E.DEP_ID = D.DEPT_ID_DEP;
```

Similarly, the column names in the select clause can be pre-fixed by aliases:

```sql
select E.EMP_ID, D.DEPT_ID_DEP  
  from employees E, departments D 
  where E.DEP_ID = D.DEPT_ID_DEP;
```

Let's say we want to see the department name for each employee:

```sql
select E.EMP_ID, D.DEP_NAME from 
  employees E, departments D
  where E.DEP_ID = D.DEPT_ID_DEP
```

However, if two tables have different column names, we don't need to use prefixes of tables. For example:

```sql
-- Full command using table prefixes:
SELECT E.F_NAME, D.DEP_NAME 
	FROM EMPLOYEES E, DEPARTMENTS D 
	WHERE E.DEP_ID = D.DEPT_ID_DEP

-- Short command, given that two tables have different column names:
SELECT F_NAME, DEP_NAME 
	FROM EMPLOYEES, DEPARTMENTS 
	WHERE DEPT_ID_DEP = DEP_ID
```

3. JOIN OPERATOR

We can explicitly use the `JOIN` operator to `JOIN` multiple tables in a single query. These include `INNER JOIN` and `OUTER JOIN` and are discussed in detail in additional lessons.


### QUIZ: Functions, Sub-Queries, Multiple Tables

## WEEK 3: Accessing Databases using Python



### Video 1: How to access DBs using Python?

**Advantages of using Python to access DB:**
* Rich ecosystem: NumPy, pandas, matplotlib, SciPy
* Ease of use
* Portable (open-source)
* Support relational DB systems
* Provide Python DB APIs
* Detailed documentation

**Notebooks:**
* Example: MATLAB notebook, Jypyter notebook, Zeppelin notebook 
* Languague of choice (python, scala, R, julia)
* Sharable
* Interactive output: HTML, images, video
* Big data integration: leverage big data tools such as: Apache Spark from Python, R, and Scala,
and explore that same data with pandas, scikit-learn, ggplot2, and TensorFlow

**SQL API:** consists of a library of function calls as an application programming interface (API) for the DBMS

**API's used by popular SQL-based DBMS systems**

-----------------------------------------------------
Application or DB 			| SQL API
-----------------------------------------------------
MySQL						| MySQL Connector/Python
PostgreSQL					| psycopg2
IBM DB2 					| ibm_db
SQL Server 					| dblib API
DB access for MS Windows 	| ODBC
Oracle 						| OCI
Java 						| JDBC
MongoDB						| PyMongo
-----------------------------------------------------


### Video 2: Writing code using DB-API


Jupyter notebook <------ DB API calls ------> DBMS
(python programs)

**Concepts of Python DB API**

Connection objects:
* DB connections
* Manage transactions

Cursor objects: scan throuh the results of a DB
* DB queries

1. Connection methods
* `cursor()`: returns a new cursor object using the connection
* `commit()`: commits any pending transation to the DB
* `rollback()`: causes the DB to roll back to the start of any pending transaction
* `close()`: close a DB connection

These objects represent a database cursor which is used to manage the content of a fetch operation:

`callproc()`, `execute()`, `executemany()`, 
`fetchone()`, `fetchmany()`, `fetchall()`,
`nextset()`, `Arraysize()`, `close()`


* Cursors created from the same connection are not isolated i.e. any changes done to the database by a cursor are immediately visible by the other cursors
* Cursors created from different connections can or cannot be isolated depending on how the transaction support is implemented
* A database cursor is a control structure that enables traversal over the records in a database
* It behaves like a file name or file handle in a programming language. 

------------------------------
| Your 			-------------|  
| Application 	| open()	 |---------- 		   ------------
|				| execute()	 | cursor  | <-------> | Database |
|				| fetchall() |----------		   ------------
|				| close()	 |
|				-------------|
-----------------------------

* Just as a program opens a filed accesses files contents, it opens a cursor to gain access to the query results
* Similarly, the program closes a file to end its access and closes a cursor to end access to the query results
* Another similarity is that just as file handle keeps track of the program's current position within an open file, a cursor keeps track of the program's current position within the query results

```python
from dbmodule import connect 

# Create connection object
Connection = connect('database_name', 'username', 'password')

# Create a cursor object
Cursor = connection.cursor()

# Run queries
Cursor.execute('select * from my_table')
Results = Cursor.fetchall()

# Free resources
Cursor.close()
Connection.close()


```

Summary: 
* A database cursor is a control structure that enables traversal over the records in a database
* A database cursor is like a file handle in that it enables you to scan through a result set of a query


### Video 3: Connecting to a DB using `ibm_db` API

* The `ibm_db` API provides a variety of useful Python functions for accessing and manipulating data in an IBM data server database, including functions for
	* connecting to a database
	* repairing and issuing sequel statementsfetching rows from result sets
	* calling stored procedures
	* committing and rolling back transactions
	* handling errors and retrieving metadata

* The `ibm_db` API uses the IBM Data Server Driver for ODBC, and CLI APIs to connect to `IBM DB2`, and `Informix`


```python
import ibm_db

'''
Connecting to the DB2 warehouse requires the following information: a driver name, a database name, a host DNS name or IP address, a host port, a connection protocol, a user ID, and a user password. 
'''

dsn_driver 	 = "IBM DB2 ODBC DRIVER"
dsn_database = "BLUDB" 	# e.g., "BLUDB"
dsn_hostname = "dashdb-entry-yp-dal10-01.services.dal.bluemix.net"
dsn_port 	 = "50001"
dsn_protocol = "TCPIP"
dsn_uid 	 = "dahs104434"
dsn_pwd 	 = "ABCXYZ"

'''
Example of creating a DB2 warehouse database connection in python:
'''

# Create DB connection
dsn = (
	"DRIVER = {{IBM DB2 ODBC DRIVER}};"
	"DATABASE = {0};"
	"HOSTNAME = {1};"
	"PORT = {2};"
	"PROTOCOL = TCPIP"
	"UID = {3};"
	"PWD = {4};"
	).format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd) 

try:
	conn = ibm_db.connect(dsn, "", "")
	print("Connected!")
except:
	print("Unable to connect to database")

# Close the DB connection
ibm_db.close(conn)

```

In the above example:
* We create a connection object DSN, which stores the connection credentials
* The connect function of the ibm_db API will be used to create a non persistent connection
* The DSN object is passed as a parameter to the connection function
* If a connection has been established with the database, then the code returns connected, as the output otherwise, the output will be unable to connect to database
* Then we free all resources by closing the connection
* Remember that it is always important to close connections so that we can avoid unused connections taking up resources

**Summary:**
* `ibm_db` API library provides SQL APIs for Python applications
* `ibm_db` API library can only be used to connect to certain IBM databases like Db2 and Db2 Warehouse on Cloud.
* `ibm_db` API includes functions for connecting to a database, repairing and issuing sequel statements, fetching rose from result sets.
* `ibm_db` API includes functions for calling stored procedures, committing and rolling back transactions, handling errors and retrieving metadata.
* `ibm_db` API provides a variety of useful Python functions for accessing and manipulating data in an IBM data server database.


### Lab 1: Connecting to a DB instance


### Video 4: Creating tables, inserting and querying data

* To create a table, we use the `ibm_db.exec_immediate()` function, with following parameters:
	* Connection: A valid database connection resource that is returned from the ibm_dbconnect or ibm_dbpconnect function
	* Statement: A string that contains the sequel statement
	* Options: Optional parameter that includes a dictionary that specifies the type of cursor to return for results sets

```python
# Python code to create a table
stmt  = ibm_db.exec_immediate(conn,
	"CREATE TABLE Trucks(
		serial_no VARCHAR(20) PRIMARY KEY NOT NULL,
		model VARCHAR(20) NOT NULL,
		manufacturer VARCHAR(20) NOT NULL,
		Engine_size VARCHAR(20) NOT NULL,
		Truck_Class VARCHAR(20) NOT NULL)")

# Python code to insert data into the table
stmt  = ibm_db.exec_immediate(conn,
	"INSERT INTO Trucks(serial_no, model, manufacturer, Engine_size, Truck_Class)
		VALUE('A1234', 'Lonestar', 'International Trucks', 'Cummins ISX15', 'Class8');")

# Insert more rows to the table
stmt = ibm_db.exec_immediate(conn,
	"INSERT INTO Trucks(serial_no, model, manufacturer, Engine_size, Truck_Class)
		VALUE('B5432', 'Volvo VN', 'Volvo Trucks', 'Volvo D11 ', 'Heavy Duty Tractor Class 8');")

stmt = ibm_db.exec_immediate(conn,
	"INSERT INTO Trucks(serial_no, model, manufacturer, Engine_size, Truck_Class)
		VALUE('C5674', 'Kenworth W900', 'Kenworth Truck Company', 'Caterpillar C9', 'Class8');")

```

Result: TABLE `Trucks`
---------------------------------------------------------------------------------------------------------
Serial No 	| Model 		| Manufacturer 				| Engine Size 		| Class
---------------------------------------------------------------------------------------------------------
A1234 		| Lonestar		| International Trucks 		| Cummins ISX15 	| Class8
B5432 		| Volvo VN 		| Volvo Trucks 				| Volvo D11 		| Heavy Duty Tractor Class 8
C5674 		| Kenworth W900	| Kenworth Truck Company	| Caterpillar C9	| Class8
--------------------------------------------------------------------------------------------------------- 


```python
# Python code to query data
stmt = ibm_db.exec_immediate(conn, "SELECT * FROM Trucks")
ibm_db.fetch_both(stmt)

'''
Output:
{
	0:'A1234',
	1:'Lonestar',
	'MANUFACTURER': 'International Trucks',
	3: 'Cummins ISX15',
	....
}
'''
```
 
Using `Pandas`:

```python
import pandas as pd
import ibm_db_dbi

pconn = ibm_db_dbi.Connection(conn)
df = pd.read_sql('SELECT * FROM Trucks', pconn)
df
```


### Lab 2: Creating tables, inserting and querying data


### Video 5: Analyzing data with Python

Four steps involed in loading data into a table:
* Source
* Target
* Define 
* Finalize.

Example: Data set obtained from the nutritional facts for McDonald's menu from Kaggle.

```python
stmt = ibm_db.exec_immediate(conn, "SELECT count(*) FROM MCDONALS_NUTRITION") # Get the nb of rows
ibm_db.fetch_both(stmt)

'''
Out: {0: '260', '1': '260'}
'''
# Using Pandas:
import pandas as pd
import ibm_db_dbi

pconn = ibm_db_dbi.Connection(conn)
df = pd.read_sql('SELECT count(*) FROM MCDONALS_NUTRITION', pconn)
df.head() 	# View 5 first rows
df.describe(include = 'all') 	# View summary of the table: count, unique, top, freq, mean, std, etc.
```

Analyse data: Which food item has maximum sodium content?

**Facts:** 
* Sodium controls fluid balance in our bodies, and maintains blood volume and blood pressure
* Eating too much sodium may raise blood pressure and cause fluid retention, which could lead to swelling of the legs, and feet, or other health issues
* A common target is to eat less than 2,000 milligrams of sodium per day

```python
import matplotlib.pylot as plt
# matplotlib inline # for notebook
import seaborn as sns 

# Categorical scatterplots

plot = sns.swarmplot(x="Catergory", y="Sodium", data=df) # data df is a dataframe
plt.setp(plot.get_xticklabels(), rotation=70)
plt.title('Sodium Content')
plt.show()
```


				Sodium Content
		^
Sodium 	|				* <-- Highest sodium value
		|	**
		|	**			*	
		|	***			***
		|	**			*			**
		|	*						*
		----------------------------------->
			Breakfast	Beef&Pork	Salads
					Category


Get the maximum sodium content

```python
# Code 1:
df['Sodium'].describe()
''' 
Output:
count 	260.0000
mean
...
max 	3600.000 <- HERE
'''

# Explore the row associated with the maximum sodium variable
df['Sodium'].idxmax() 	# Output: 82

# Find the item name associated with the row id = 82
df.at[82, 'Item'] 		# Output: 'Chicken McNuggets (40 piece)'

```

Further data exploration using visualizations

1. Show the correlation between two variables: Total Fat and Protein
* Correlation is a measure of association between two variables, and has a value of between -1 and +1 

```python
import matplotlib.pylot as plt
# matplotlib inline # for notebook
import seaborn as sns 

plot = sns.jointplot(x="Protein", y="Total Fat", data=df)
plot.show()

```
				|
				| |			<-- Histogram of Protein
				| | | : .
			--------------------------------------------
			|			| pearsonr = 0.81, p = 3.7e-61	|
			|			--------------------------------|
			|											|
			|											|
			|							* <-- Outlier 	|
			| 		***									|
Total Fat 	|		*****								|*	
			|	   *******								|**
			|	   ******								|****
			|		***									|******	 <-- Histogram of Total Fat
			|											|	
			---------------------------------------------
							Protein

**Comments:**
* We see that the points on the scatter plot are closer to a straight line in the positive direction
* So we have a positive correlation between the two variables
* On the top right corner of the scatter plot, we have the values of the Pearson correlation, 0.81 and the significance of the correlation denoted as P, which is a good value that shows the variables are certainly correlated
* The plot also shows two histograms: one on the top is that of the variable protein, and one on the right side is that of the variable total fat
* We also noticed that there is a point on the scatter plot outside the general pattern. This is a possible outlier

2. Using box plots:
* Box plots are charts that indicate the distribution of one or more variables
* The box in a box plot captures the middle 50 percent of data
* Lines and points indicate possible skewness and outliers.

```python
import matplotlib.pylot as plt
# matplotlib inline # for notebook
import seaborn as sns 

plot = sns.set_style("whitegrid")
ax = sns.boxplot(x = df["Sugars"])
plot.show()

```

^
|								outliers	
|	---------					 |
|	|		|				|	 v	
|---------------------------- 	*** 
|	|		|				|
|	---------
|
-----------------------------------> 
0     20  40  60
				Sugars


**Comments:**
* Average values of sugar and food items around 30 grams
* We also notice a few outliers that indicate food items with extreme values of sugar
* There exists food items in the data set that have sugar content of around 128 grams
* Candies maybe among these high sugar content food items on the menu




### Lab 3: Analyzing a real world data set

Few notes and tips for working with real world data sets:

1. CSV files
In many cases data sets are made available as `.CSV` files which contain data values separated by commas instead of spaces 

2. Column names
Sometimes the first row of the CSV file contains the name of the column. If you are loading the data into the database using the visual LOAD tool in the console, ensure the "Header in first row" is enabled. This will map the first row into column names, and rest of the rows into data rows in the table.

3. Querying column names with mixed (upper and lower) case
In the example above we loaded the csv file using default column names. This resulted in column names having mixed case. If you try to run the following query you will get an error:

```sql
select id from dogs
```

This is because the database parser assumes uppercase names by default. To specify the mixed case column name, use double quotes (i.e. " " but not single quotes ' ') around the column name:

```sql
select "Id" from DOGS
```

4. Querying columns names with spaces and other characters
If in a csv file the name of the column contained spaces, e.g. "Name of Dog", by default the database may map them to underscores, i.e.: "Name_of_Dog". Other special characters like brackets may also get mapped to underscores. Therefore when you write a query, ensure you use proper case within quotes and substitute special characters to underscores:

```sql
select "Id", "Name_of_Dog", "Breed__dominant_breed_if_not_pure_breed_" from dogs
```

5. Using quotes in Jupyter notebooks

You may be issuing queries in a notebook by first assigning them to Python variables. In such cases, if your query contains double quotes (e.g. to specify mixed case column name), you could differentiate the quotes by using single quotes ' ' for the Python variable to enclose the SQL query, and double quotes " " for the column names.

```sql
selectQuery = 'select "Id" from dogs'
```

What if you need to specify single quotes within the query (for example to specify a value in the where clause. In this case you can use backslash \ as the escape character:

```sql
selectQuery = 'select * from dogs where "Name_of_Dog"=\'Huggy\' '
```

6. Splitting queries to multiple lines in notebooks

If you have very long queries (such as join queries), it may be useful to split the query into multiple lines for improved readability. In Python notebooks you can use the backslash \ character to indicate continuation to the next row:

```sql
selectQuery = 'select "Id", "Name_of_Dog", \ 
	"Breed__dominant_breed_if_not_pure_breed_" from dogs \                         
	where "Name_of_Dog"=\'Huggy\''          
```

If using `SQL magic`:

```sql
%sql select "Id", "Name_of_Dog", \
	"Breed__dominant_breed_if_not_pure_breed_" from dogs \                         
		where "Name_of_Dog"='Huggy'      
```

You can also use `%%sql` in the first line of the cell which tells the interpretor to treat rest of the cell as SQL:

```sql
%%sql 
select "Id", "Name_of_Dog", 
	"Breed__dominant_breed_if_not_pure_breed_" from dogs                          
		where "Name_of_Dog"='Huggy'     
```

7. Restricting number of rows retrieved

A table may contain thousands or millions of rows but you may only want to just sample some data sample or look at just a few rows to see what kind of data the table contains. You may be tempted to just do "select * from tablename", retrieve the results in a Pandas's dataframe and do a `head()` on it. But doing so may take a long time for a query to run. Instead you can limit the result set by using "LIMIT" clause. For example use the following query to retrieve just the first 10 rows:

```sql
select * from census_data LIMIT 10
```

8. Getting a list of tables in the database

Sometimes your database may contain several tables and you may not remember the correct name. For example you may wonder whether the table is called `"DOG"`, `"DOGS"` or `"FOUR_LEGGED_MAMMALS"`. Or you may have created several tables with similar names e.g. `"DOG1"`, `"DOG_TEST"`, "DOGTEST1" but want to check which was the last one you created. Database systems typically contain system or catalog tables from where you can query the list of tables and get their attributes. In `Db2` its called `"SYSCAT.TABLES"` (in `SQL Server` its `"INFORMATION_SCHEMA.TABLES"`, and in Oracle: `ALL_TABLES` or `USER_TABLES`). To get a list of tables in a `Db2` database you can run the following query:

```sql
select * from syscat.tables
```

The above will return too many tables including system tables so it is better to filter the result set as follows (ensure you replace `"QCM54854"` with your `Db2` username):

```sql
select TABSCHEMA, TABNAME, CREATE_TIME from syscat.tables where tabschema='QCM54853'
```

FYI, in MySQL you can simply do `"SHOW TABLES"`.


9. Getting a list of columns in a table

If you can't recall the exact name of a column, or want to know its attributes like the data type you can issue a query like the following for Db2:

```sql
select * from syscat.columns where tabname = 'INSTRUCTORS'
-- or:
select distinct(name), coltype, length from sysibm.syscolumns where tbname = 'INSTRUCTORS'
```

FYI, In MySQL you can run `"SHOW COLUMNS FROM TABNAME"`.





### QUIZ: Database access from Python

1. Question 1
A database cursor is a control structure that enables traversal over the records in a database. -> TRUE

2. Question 2
The ibm_db API provides a variety of useful Python functions for accessing and manipulating data in an IBM data server like Db2. Is this statement True or False?
True


* The `ibm_db` API provides a variety of useful Python functions for accessing and manipulating data in an IBM data server database, including functions for
	* connecting to a database
	* repairing and issuing sequel statementsfetching rows from result sets
	* calling stored procedures
	* committing and rolling back transactions
	* handling errors and retrieving metadata

3. Question 3
A Dataframe represents a tabular, spreadsheet-like data structure containing an ordered collection of columns, each of which can be a different value type. Indicate whether the following statement is True or False:

A pandas dataframe in Python can be used for storing the result set of a SQL query.
True

4. Question 4
Which of the following statement(s) about Python is/are NOT correct (i.e. False)?

The Python ecosystem is very rich and provides easy to use tools for data science.
Due to its proprietary nature, database access from Python is not available for many databases. -> FALSE
There are libraries and APIs available to access many of the popular databases from Python.
Python is a popular scripting language for connecting and accessing databases.


5. Question 5
To query data from tables in database a connection to the database needs to be established. Which of the following is NOT required to establish a connection with a relational database from a Python notebook:

An SQL or Database API
Username, Password, and Database name
Table and Column names <- THIS



## WEEK 4: Course Assignment

**Instruction:** Load data to `IBM Db2`
* Go to your IBM Cloud dashboard: https://console.bluemix.net/dashboard/apps
* Then click on the Db2 service and click on Open Console button. Once Db2 console opens you can select LOAD from the menu.
* For detailed instructions to launch Db2 console refer to Week 1: Hands on Lab: Composing and Running basic SQL queries
