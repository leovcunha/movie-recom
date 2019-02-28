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
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.PlusAnonymousConcurrentUserDataModel;

@Service
public class MovieRecommender {
    
   private UserBasedRecommender recommender;
   private Logger logger;
   
   
   public MovieRecommender() {
       try{         
         logger = LoggerFactory.getLogger(MovieRecommender.class); 
         
         
         
         //Creating data model
         DataModel datamodel = new FileDataModel(new File("/projects/movie-recom/data/ratings_small.csv")); //data
         
         PlusAnonymousConcurrentUserDataModel plusModel = new PlusAnonymousConcurrentUserDataModel(datamodel, 10);
         //Creating UserSimilarity object.
         UserSimilarity usersimilarity = new PearsonCorrelationSimilarity(plusModel);
      
         //Creating UserNeighbourHHood object.
         UserNeighborhood userneighborhood = new ThresholdUserNeighborhood(0.5, usersimilarity, plusModel);

         //Create UserRecomender
         recommender = new GenericUserBasedRecommender(plusModel, userneighborhood, usersimilarity);   
         
      } catch(Exception e){
          this.logger.info(e.toString());
      }
   }
   
   public Map<Long, Float> getRecommendations(Map<Long, Float> userPreferences) {
       
     List<RecommendedItem> recommendations; 
     Map<Long, Float> res = new HashMap<>();
     GenericUserPreferenceArray tempPrefs = new GenericUserPreferenceArray(userPreferences.size());
     PlusAnonymousConcurrentUserDataModel plusModel =
           (PlusAnonymousConcurrentUserDataModel) recommender.getDataModel();     
     long newUserID = plusModel.takeAvailableUser(); 
     
     try {
        int i = 0;
        for(Map.Entry<Long, Float> entry : userPreferences.entrySet()) {
            Long key = entry.getKey();
            Float value = entry.getValue();
        
            tempPrefs.setUserID(i, newUserID);
            tempPrefs.setItemID(i, key);
            tempPrefs.setValue(i, value);
        
            i++;
        }
        this.logger.info("{}", tempPrefs);
        // Add the temporaly preferences to model


        plusModel.setTempPrefs(tempPrefs, newUserID);         
        this.logger.info("{}", plusModel.getPreferencesFromUser(newUserID));

        recommendations = recommender.recommend(newUserID, 10);
		plusModel.releaseUser(newUserID);
		 
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
