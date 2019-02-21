package com.movierec.recommender;

import java.io.File;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class MovieRecommender {
    
   private DataModel datamodel;
   private UserSimilarity usersimilarity;
   private UserNeighborhood userneighborhood;
   private UserBasedRecommender recommender;
   
   public MovieRecommender() {
                
         Logger logger = LoggerFactory.getLogger(MovieRecommender.class); 
         
         //Creating data model
         DataModel datamodel = new FileDataModel(new File("/projects/movie-recom/data/ratings_small.csv")); //data
         
         //Creating UserSimilarity object.
         UserSimilarity usersimilarity = new PearsonCorrelationSimilarity(datamodel);
      
         //Creating UserNeighbourHHood object.
         UserNeighborhood userneighborhood = new ThresholdUserNeighborhood(0.5, usersimilarity, datamodel);
      
         //Create UserRecomender
         UserBasedRecommender recommender = new GenericUserBasedRecommender(datamodel, userneighborhood, usersimilarity);    
   }
  
   public static List<RecommendedItem> getRecommendations(int userId, int number){
      try{
        
         List<RecommendedItem> recommendations = recommender.recommend(userId, number);
			
         for (RecommendedItem recommendation : recommendations) {
            logger.info("recommendation: {}", recommendation.getItemID());
         }
         return recommendations;
      
      } catch(Exception e){
          throw e;
      }
      
   }
  
    
}
