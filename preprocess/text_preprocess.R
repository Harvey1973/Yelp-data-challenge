library(rjson)
library(textcat)
file_review = "C:/Users/Harvey/Desktop/Yelp_data_set/yelp_review.json"
con = file(file_review, "r") 
input <- readLines(con, 1000000L) 
reviews <- as.data.frame(t(sapply(input,fromJSON)))
row.names(reviews) <- seq(1, nrow(reviews))


keeps <- c("stars", "text","business_id")
reviews = reviews[keeps]


reviews$stars=as.numeric(reviews$stars)
reviews$text=as.character(reviews$text)

file_business = "C:/Users/Harvey/Desktop/Yelp_data_set/yelp_business.json"
con2 = file(file_business, "r") 
input2 <- readLines(con2, 500000L) 
business <- as.data.frame(t(sapply(input2,fromJSON)))
row.names(business) <- seq(1, nrow(business))

keeps_business <- c("business_id", "name", "city", "stars", "review_count", "categories")
business = business[keeps_business]

business$is_restaurant = grepl("Restaurants", business$categories)
business = subset(business, is_restaurant == TRUE)
business[144:147,]
restaurant_reviews <- reviews[which(reviews$business_id %in% business$business_id),]

restaurant_reviews = subset(restaurant_reviews, stars != 3)
restaurant_reviews$positive = as.factor(restaurant_reviews$stars > 3)

restaurant_reviews$language = textcat(restaurant_reviews$text)
restaurant_reviews = subset(restaurant_reviews, language =='english')

restaurant_reviews$business_id = NULL
write.csv(restaurant_reviews, file = "restuarant_review.csv",row.names=FALSE)

restaurant_reviews$stars = NULL