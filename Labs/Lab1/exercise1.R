EPI_data <- read.csv("epi2024results06022024.csv")

attach(EPI_data)

EPI.new

NAs <- is.na(EPI.new)

EPI.new.noNAs <- EPI.new[!NAs]

summary(EPI.new)

fivenum(EPI.new,na.rm=TRUE)

stem(EPI.new)

hist(EPI.new)

#hist(EPI.new, seq(20., 80., 1.0), prob=TRUE)

#lines(density(EPI.new,na.rm=TRUE,bw=1.))

#rug(EPI.new)

#boxplot(EPI.new, APO.new)

#hist(EPI.new, seq(20., 80., 1.0), prob=TRUE)

#lines (density(EPI.new,na.rm=TRUE,bw=1.))

#rug(EPI.new)

hist(EPI.new, seq(20., 80., 1.0), prob=TRUE)

lines (density(EPI.new,na.rm=TRUE,bw="SJ")) 

rug(EPI.new)

x<-seq(20,80,1) 
q<- dnorm(x,mean=42, sd=5,log=FALSE) 
lines(x,q)
lines(x,.4*q) 
q<-dnorm(x,mean=65, sd=5,log=FALSE) 
lines(x,.12*q) 
