function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end

function shuffle(array)
    math.randomseed(os.time())
    math.random();math.random();math.random()
    local counter = #array
    while counter > 1 do
        local index = math.random(counter)
        swap(array, index, counter)
        counter = counter - 1
    end
end

function write(path, tab)
    local file = assert(io.open(path, "w"))
    for i=1,#tab do
        file:write(tab[i])
        file:write('\n')
    end
    file:close()
end

function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

function initialize(box)

	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

	-- each stimulation sent that gets rendered by Display Cue Image box 
	-- should probably have a little period of time before the next one or the box wont be happy
	pre_baseline_duration = 1.0
	baseline_duration = 3.0
	post_baseline_duration = 1.0
	cross_duration = 0.5
	post_cross_duration = 0.75
	display_cue_duration = 0.3
	post_cue_duration = 0.4
	rest_duration = 0.4
	post_end_duration = 0.4
    
    emotional_faces = {
        OVTK_StimulationId_Label_01,
        OVTK_StimulationId_Label_03,
        OVTK_StimulationId_Label_04,
        OVTK_StimulationId_Label_05,
        OVTK_StimulationId_Label_07,
        OVTK_StimulationId_Label_08,
        OVTK_StimulationId_Label_0A,
        OVTK_StimulationId_Label_0C,
        OVTK_StimulationId_Label_0D,
        OVTK_StimulationId_Label_0E,
        OVTK_StimulationId_Label_10,
        OVTK_StimulationId_Label_12,
        OVTK_StimulationId_Label_14,
        OVTK_StimulationId_Label_15,
        OVTK_StimulationId_Label_16,
        OVTK_StimulationId_Label_17,
	}
    
    neutral_faces = {
        OVTK_StimulationId_Label_02,
        OVTK_StimulationId_Label_06,
        OVTK_StimulationId_Label_09,
        OVTK_StimulationId_Label_0B,
        OVTK_StimulationId_Label_0F,
        OVTK_StimulationId_Label_11,
        OVTK_StimulationId_Label_13,
        OVTK_StimulationId_Label_18,
	}
    -- doubled number of neutral images
    neutral_faces = TableConcat(neutral_faces, neutral_faces)
    sequence = TableConcat(neutral_faces, emotional_faces)
  
    shuffle(sequence)
    write(box:get_config("${Player_ScenarioDirectory}") .. os.date("/signals/csv/order-[%Y.%m.%d-%H.%M.%S].csv"), sequence)
    box:set_filter_mode(1)
	
end

function process(box)

	local t = 0
    local n = 0
    -- local buttons = nil

	-- Delays before the trial sequence starts
	box:send_stimulation(1, OVTK_StimulationId_ExperimentStart, t, 0)
	t = t + pre_baseline_duration

	box:send_stimulation(1, OVTK_StimulationId_BaselineStart, t, 0)
	t = t + baseline_duration

	box:send_stimulation(1, OVTK_StimulationId_BaselineStop, t, 0)
	t = t + post_baseline_duration
       
    write(box:get_config("${Player_ScenarioDirectory}") .. os.date("/signals/csv/start.csv"), buttons)
    
    while 
    write(box:get_config("${Player_ScenarioDirectory}") .. os.date("/signals/csv/start.csv"), buttons)

	-- creates each trial
	for i = 1, #sequence do

		-- first display a cross on screen
		box:send_stimulation(1, OVTK_GDF_Start_Of_Trial, t, 0)
		box:send_stimulation(1, OVTK_GDF_Cross_On_Screen, t, 0)
		t = t + cross_duration

		-- Clear cross. 
		box:send_stimulation(1, OVTK_StimulationId_Label_19, t, 0)
		t = t + post_cross_duration
		
		-- display cue
		box:send_stimulation(1, sequence[i], t, 0)
		t = t + display_cue_duration

		-- clear cue. 
		box:send_stimulation(1, OVTK_StimulationId_Label_19, t, 0)
        t = t + post_cue_duration
       
        -- wait for input.
 
        while box:get_stimulation_count(1) < i do
            box:sleep()
        end

        
        -- for stim = 1, box:get_stimulation_count(1) do
            -- -- gets the received stimulation
            -- identifier, timestamp, duration = box:get_stimulation(1, 1)
            -- -- logs the received stimulation
            -- box:log("Trace", string.format("At time %f on input %i got stimulation id:%s date:%s duration:%s", t, 1, identifier, timestamp, duration))
            -- -- discards it
            -- box:remove_stimulation(1, 1)
        -- end
        -- t = box:get_current_time()
        
        
        --buttons = {next=buttons, value=box:get_stimulation_count(1)}
        
        --for b = 1, box:get_stimulation_count(1) do
        --write(box:get_config("${Player_ScenarioDirectory}") .. os.date("/signals/csv/buttd-[%Y.%m.%d-%H.%M.%S].csv"), {box:get_stimulation_count(1)})
        --end
        
        --for b = 1, box:get_stimulation_count(1) do
        --   box:remove_stimulation(1, 1)
        --end
        
		-- rest period
		box:send_stimulation(1, OVTK_StimulationId_RestStart, t, 0)
		t = t + rest_duration	
		
		-- end of rest and trial
		box:send_stimulation(1, OVTK_StimulationId_Label_19, t, 0)
		box:send_stimulation(1, OVTK_StimulationId_RestStop, t, 0)
		box:send_stimulation(1, OVTK_GDF_End_Of_Trial, t, 0)
		t = t + post_end_duration	
	end

	-- send end for completeness	
	box:send_stimulation(1, OVTK_GDF_End_Of_Session, t, 0)
	t = t + 5
    
    -- save answers
    -- write(box:get_config("${Player_ScenarioDirectory}") .. os.date("/signals/csv/buttons-[%Y.%m.%d-%H.%M.%S].csv"), buttons)

	-- used to cause the acquisition scenario to stop and denote final end of file
	box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, t, 0)
	
end
