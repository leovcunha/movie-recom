package com.movierec.recommender;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.hateoas.Resource;

import static org.springframework.hateoas.mvc.ControllerLinkBuilder.linkTo;
import static org.springframework.hateoas.mvc.ControllerLinkBuilder.methodOn;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import com.movierec.SpectatorRepository;
import com.movierec.Spectator;


@RestController
@RequestMapping("/api/recommendations")
public class RecommenderController {
    
    private final SpectatorRepository repository;
    private final MovieRecommender movierecom;
    
    @Autowired
    public RecommenderController(SpectatorRepository repository, MovieRecommender movierecom) {
        this.repository = repository;       
        this.movierecom = movierecom;
    }
       
    @RequestMapping(method = RequestMethod.GET, value = "/{id}") 
    public @ResponseBody ResponseEntity<?> pushRecommendations(@PathVariable long id) {
        Map<Long, Float> recommendations; 
        
        //
        // do some intermediate processing, logging, etc. with the producers
        //
        Spectator spectator = this.repository.findById(id).get(); //Optional.get() -> return the value.
        recommendations = movierecom.getRecommendations(id, 5);
        spectator.setRecommendations(recommendations);
        repository.save(spectator);
        
        Resource<Spectator> resource = new Resource<Spectator>(spectator); 
        
        resource.add(linkTo(methodOn(RecommenderController.class).pushRecommendations(id)).withSelfRel());
        // add other links as needed

        return ResponseEntity.ok(resource); 
    }
    
    
}


