# Reading Data
EPI_data <- read.csv("epi2024results06022024.csv")

attach(EPI_data)

EPI.new

#Cleaning Data
NAs <- is.na(EPI.new)

EPI.new.noNAs <- EPI.new[!NAs]

# Exercise 1
summary(EPI.new)
fivenum(EPI.new,na.rm=TRUE)
stem(EPI.new)
hist(EPI.new)

hist(EPI.new, seq(20., 80., 1.0), prob=TRUE)

lines(density(EPI.new,na.rm=TRUE,bw=1.))

rug(EPI.new)

boxplot(EPI.new, APO.new)

hist(EPI.new, seq(20., 80., 1.0), prob=TRUE)

lines (density(EPI.new,na.rm=TRUE,bw=1.))

rug(EPI.new)

hist(EPI.new, seq(20., 80., 1.0), prob=TRUE)

lines (density(EPI.new,na.rm=TRUE,bw="SJ")) 

rug(EPI.new)

x<-seq(20,80,1) 
q<- dnorm(x,mean=42, sd=5,log=FALSE) 
lines(x,q)
lines(x,.4*q) 
q<-dnorm(x,mean=65, sd=5,log=FALSE) 
lines(x,.12*q) 

# Exercise 2

plot(ecdf(EPI.new), do.points=FALSE, verticals=TRUE) 
qqnorm(EPI.new); qqline(EPI.new) 
qqplot(rnorm(250), EPI.new, xlab = "Q-Q plot for norm dsn") 
qqline(EPI.new)

qqplot(rt(250, df = 5), EPI.new, xlab = "Q-Q plot for t dsn") 
qqline(EPI.new)


plot(ecdf(APO.new), do.points=FALSE, verticals=TRUE) 
qqnorm(APO.new); qqline(APO.new) 
qqplot(rnorm(250), APO.new, xlab = "Q-Q plot for norm dsn") 
qqline(APO.new)

qqplot(rt(250, df = 5), APO.new, xlab = "Q-Q plot for t dsn") 
qqline(APO.new)


plot(ecdf(WRS.new), do.points=FALSE, verticals=TRUE) 
qqnorm(WRS.new); qqline(WRS.new) 
qqplot(rnorm(250), WRS.new, xlab = "Q-Q plot for norm dsn") 
qqline(WRS.new)

qqplot(rt(250, df = 5), WRS.new, xlab = "Q-Q plot for t dsn") 
qqline(WRS.new)


