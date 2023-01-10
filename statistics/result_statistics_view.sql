create view statistics as 
      select 
           test_id, 
           key, 
           COUNT(*) as real_runs, 
           runs, 
           AVG(value) as mean, 
           stdev(value) as st_dev, 
           stdev(value)/sqrt(count(*)) as st_err,  
           mode(value) as mode, 
           median(value) as median, 
           avg(value)-stdev(value)/sqrt(count(*))*max(p) as conf_in1, 
           avg(value)+stdev(value)/sqrt(count(*))*max(p) as conf_in2,  
           unit
       from result, testrun, test, t_distribution  
       where df=runs-1 and  result.testrun_id = testrun.id 
         and testrun.test_id = test.id
       group by test_id, key;
