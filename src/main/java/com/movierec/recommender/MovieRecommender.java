package com.movierec.recommender;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

@Service
public class MovieRecommender {
    
   private DataModel datamodel;
   private UserSimilarity usersimilarity;
   private UserNeighborhood userneighborhood;
   private UserBasedRecommender recommender;
   private Logger logger;
   
   
   public MovieRecommender() {
       try{         
         logger = LoggerFactory.getLogger(MovieRecommender.class); 
         
         //Creating data model
         datamodel = new FileDataModel(new File("/projects/movie-recom/data/ratings_small.csv")); //data
         
         //Creating UserSimilarity object.
         usersimilarity = new PearsonCorrelationSimilarity(datamodel);
      
         //Creating UserNeighbourHHood object.
         userneighborhood = new ThresholdUserNeighborhood(0.5, usersimilarity, datamodel);
      
         //Create UserRecomender
         recommender = new GenericUserBasedRecommender(datamodel, userneighborhood, usersimilarity);   
      } catch(Exception e){
          this.logger.info(e.toString());
      }
   }
   
   public Map<Long, Float> getRecommendations(long userId, int number) {
       
     List<RecommendedItem> recommendations; 
     Map<Long, Float> res = new HashMap<>();
     
     try {
         recommendations = recommender.recommend(userId, number);
			
         for (RecommendedItem recommendation : recommendations) {
            this.logger.info("recommendation: {}", recommendation.getItemID());
            res.put(recommendation.getItemID(), recommendation.getValue());
         }
         return res;
         
     } catch(Exception e){
      this.logger.info(e.toString());
      return null;
      
      
     }  
   }
  

  
    
}
