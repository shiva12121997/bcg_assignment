import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, row_number, sum
from pyspark.sql.window import Window

class CarCrashAnalysis:
    def __init__(self, input_config):
        '''
        Initializing data from csv
        '''
        self.charges = spark.read.csv(input_config['charges'], header = True)
        self.damages = spark.read.csv(input_config['damages'], header = True)
        self.endorse = spark.read.csv(input_config['endorse'], header = True)
        self.primary = spark.read.csv(input_config['primary'], header = True)
        self.restrict = spark.read.csv(input_config['restrict'], header = True)
        self.units = spark.read.csv(input_config['units'], header = True)

    def analysis1(self, output_fn):
        '''
        Gives the number of crashes (accidents) in which number of persons killed are male
        Parameter: output file name
        Returns: counts of dataframe
        '''
        df = self.primary.filter((col('PRSN_INJRY_SEV_ID') == 'KILLED') & (col('PRSN_GNDR_ID') == 'MALE'))
        df.toPandas().to_csv(output_fn)
        return df.count()

    def analysis2(self, output_fn):
        '''
        Gives the number of two wheelers which are booked for crashes
        Parameter: output file name
        Return: counts of dataframe 
        '''
        df = self.units.filter(col('VEH_BODY_STYL_ID') == 'MOTORCYCLE')
        df.toPandas().to_csv(output_fn)
        return df.count()
    
    def analysis3(self, output_fn):
        '''
        Gives the name of state which has highest number of accidents in which females are involved
        Parameter: output file name
        Return: State Name
        '''
        df = self.primary.filter(col('PRSN_GNDR_ID') == 'FEMALE').groupBy(col('DRVR_LIC_STATE_ID')).agg(count('*').alias('cnt')). \
            orderBy(col('cnt'), ascending = False).limit(1)
        df.toPandas().to_csv(output_fn)
        return df.select('DRVR_LIC_STATE_ID').collect()[0]
    
    def analysis4(self, output_fn):
        '''
        Gives the top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
        Parameter: output file name
        Return: list of top 5th to 15th VEH_MAKE_IDs
        '''
        df = self.units.withColumn('TOTAL_INJ_DEATH_CNT', col('TOT_INJRY_CNT') + col('DEATH_CNT'))
        df = df.groupBy('VEH_MAKE_ID').agg(sum('TOTAL_INJ_DEATH_CNT').alias('AGG_TOTAL_INJ_DEATH_CNT')).\
                orderBy(col('AGG_TOTAL_INJ_DEATH_CNT').desc()).filter(col('VEH_MAKE_ID') != 'NA')
        
        window_spec = Window.orderBy(col('AGG_TOTAL_INJ_DEATH_CNT').desc())
        df = df.withColumn('rn', row_number().over(window_spec)).filter((col('rn') >= 5) & (col('rn') <= 15))

        df.toPandas().to_csv(output_fn)
        res = [i[0] for i in df.select(col('VEH_MAKE_ID')).collect()]
        return res

    def analysis5(self, output_fn):
        '''
        Gives the top ethnic user group of each unique body style that are involved in crashes
        Parameter: output file name
        Return: df of ethnic user group
        '''
        window_spec = Window.partitionBy(col('VEH_BODY_STYL_ID')).orderBy(col('count').desc())
        df = self.primary.join(self.units, on = ['CRASH_ID'], how = 'inner').select(col('PRSN_ETHNICITY_ID'), col('VEH_BODY_STYL_ID')). \
                groupBy('VEH_BODY_STYL_ID', 'PRSN_ETHNICITY_ID').count()
        df = df.withColumn('rn', row_number().over(window_spec)).filter((col('rn') == 1) & (~col('VEH_BODY_STYL_ID').isin('NA')))
        df.toPandas().to_csv(output_fn)
        return df.select('VEH_BODY_STYL_ID', 'PRSN_ETHNICITY_ID').collect()

    def analysis6(self, output_fn):
        '''
        Gives the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash
        Parameter: output file name
        Return: list of top 5th to 15th VEH_MAKE_IDs
        '''
        df = self.primary.join(self.units, on = ['CRASH_ID'], how = 'inner').select('CONTRIB_FACTR_1_ID', 'CONTRIB_FACTR_2_ID', 'DRVR_ZIP')
        df = df.filter(col('CONTRIB_FACTR_1_ID').contains('ALCOHOL') | col('CONTRIB_FACTR_2_ID').contains('ALCOHOL')). \
                dropna(subset=['DRVR_ZIP']).groupBy(col('DRVR_ZIP')).count().orderBy(col('count').desc()).limit(5)
        df.toPandas().to_csv(output_fn)
        res = [i[0] for i in df.select(col('DRVR_ZIP')).collect()]
        return res

    def analysis7(self, output_fn):
        '''
        Gives the Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
        Parameter: output file name
        Return: Count of distinct Crash ID
        '''
        df = self.units.join(self.damages, on = ['CRASH_ID'], how = 'inner').\
                filter(col('DAMAGED_PROPERTY').contains('NONE')).\
                filter(
                        (col('VEH_DMAG_SCL_2_ID') > 'DAMAGED 4') & (~col('VEH_DMAG_SCL_2_ID').isin('INVALID VALUE', 'NA', 'NO DAMAGE')) |
                        (col('VEH_DMAG_SCL_1_ID') > 'DAMAGED 4') & (~col('VEH_DMAG_SCL_1_ID').isin('INVALID VALUE', 'NA', 'NO DAMAGE'))
                    ).\
                filter(col('FIN_RESP_TYPE_ID') == 'PROOF OF LIABILITY INSURANCE')
        df.select('CRASH_ID').distinct().toPandas().to_csv(output_fn)
        return df.select('CRASH_ID').distinct().count()

    def analysis8(self, output_fn):
        '''
        Gives the list of Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences
        Parameter: output file name
        Return: list of Vehicle Makes
        '''
        top_vehicle_colour = self.units.groupBy(col('VEH_COLOR_ID')).count().orderBy(col('count').desc()).\
                                filter(col('VEH_COLOR_ID') != 'NA').limit(10)
        top_vehicle_colour_li = [i[0] for i in top_vehicle_colour.collect()]

        top_states = self.units.groupBy(col('VEH_LIC_STATE_ID')).count().orderBy(col('count').desc()).\
                        filter(col('VEH_LIC_STATE_ID') != 'NA').limit(25)
        top_states_li = [i[0] for i in top_states.collect()]

        df = self.units.filter((col('VEH_COLOR_ID').isin(top_vehicle_colour_li)) & (col('VEH_LIC_STATE_ID').isin(top_states_li))).\
                join(self.charges, on = ['CRASH_ID'], how = 'inner').filter(col('CHARGE').contains('SPEED')).\
                join(self.primary, on = ['CRASH_ID'], how = 'inner').filter(col('DRVR_LIC_TYPE_ID').contains('DRIVER')).\
                groupBy('VEH_MAKE_ID').count().orderBy(col('count').desc()).limit(5)
        
        df.toPandas().to_csv(output_fn)

        return [i[0] for i in df.select('VEH_MAKE_ID').collect()]

if __name__ == '__main__':
    # Creating SparkSession from builder
    spark = SparkSession.builder.appName('CarCrashAnalysis').getOrCreate()
    
    # Loading config json File
    data = json.load(open('config.json'))
    output = data['output']

    analysis = CarCrashAnalysis(data['input'])
    
    #Find the number of crashes (accidents) in which number of persons killed are male?
    print('1. Result: ' + str(analysis.analysis1(output['out1'])))

    # How many two wheelers are booked for crashes? 
    print('2. Result: ' + str(analysis.analysis2(output['out2'])))

    # # Which state has highest number of accidents in which females are involved? 
    print('3. Result: ' + str(analysis.analysis3(output['out3'])))

    # # Which are the Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
    print('4. Result: ' + str(analysis.analysis4(output['out4'])))

    # # For all the body styles involved in crashes, mention the top ethnic user group of each unique body style  
    print('5. Result: ' + str(analysis.analysis5(output['out5'])))
    
    # # Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)
    print('6. Result: ' + str(analysis.analysis6(output['out6'])))
    
    # # Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
    print('7. Result: ' + str(analysis.analysis7(output['out7'])))
    
    # # Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences (to be deduced from the data)
    print('8. Result: ' + str(analysis.analysis8(output['out8'])))

    spark.stop()