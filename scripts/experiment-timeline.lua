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

function sleep(a) 
    local sec = tonumber(os.clock() + a); 
    while (os.clock() < sec) do 
    end 
end

-- TODO: this function can be simpler and more modular
function is_answer_good(task_FET, task_Z, face_id, answer)
    
    labels = {}
    if task_FET then
        -- neutral labels
        labels = {2, 6, 9, 11, 15, 17, 19, 24}
    else
        -- men labels
        labels = {7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21}
    end
    
    -- check if face_id in labels
    found = false
    for i = 1, #labels do
        if face_id == 33024 + labels[i] then
            found = true
            break
        end 
    end
    
    -- if primary button was pressed
    primary_answer = false
    if (task_Z and answer == 2) or (taskZ == false and answer == 4) then
        primary_answer = true
    end
    
    if found then
        if task_Z and answer == 4 then return true end         
        if task_Z == false and answer == 2 then return true end
    else
        if task_Z and answer == 2 then return true end         
        if task_Z == false and answer == 4 then return true end
    end

    return false
end

function initialize(box)
    
	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

	-- each stimulation sent that gets rendered by Display Cue Image box 
	-- should probably have a little period of time before the next one or the box wont be happy
	cross_duration = 0.5
	post_cross_duration = 0.75
	display_cue_duration = 0.3
	rest_duration = 1.2
    
    inter_block_rest = 15
    block_size = 64
    experiment_units = 8
    
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
    TableConcat(neutral_faces, neutral_faces)
    
    sequence = {}
    for i = 1, experiment_units do
        TableConcat(sequence, neutral_faces)
        TableConcat(sequence, emotional_faces)
    end
    shuffle(sequence)
    box:log("Info", string.format("%i faces in the task", #sequence))
    
    -- get task type
    task_FET = false
    if box:get_setting(2) == "FET" then task_FET = true end
        
    -- get primary button
    task_Z = false
    if box:get_setting(3) == "Z" then task_Z = true end
	
    -- process() will be event-driven
    box:set_filter_mode(1)
    
end

function uninitialize(box)
    box:log("Info", "uninitialized")
end

function wait_until(box, t)
    while box:get_current_time() < t do
        box:sleep()
    end
end

function wait_for_sec(box, t)
    wait_until(box, box:get_current_time() + t)
end

function wait_for_id(box, id)
    key_id = 0
    while box:keep_processing() and key_id~=id do
        
        for stimulation = 1, box:get_stimulation_count(1) do
            key_id, timestamp, duration = box:get_stimulation(1, 1)
            box:remove_stimulation(1, 1)           
        end
   
        box:sleep()
    end
    
    -- clear input buffer
    for stimulation = 1, box:get_stimulation_count(1) do
        box:remove_stimulation(1, 1)
    end
end

function process(box)
    
    -- show experiment instructions
    box:send_stimulation(1, OVTK_StimulationId_Label_1C, box:get_current_time(), 0)
    wait_for_id(box, 2)
    
    box:send_stimulation(1, OVTK_StimulationId_Label_1D, box:get_current_time(), 0)
    wait_for_id(box, 2)
     
    box:send_stimulation(1, OVTK_StimulationId_Label_1F, box:get_current_time(), 0)
    wait_for_id(box, 2)
	
    box:send_stimulation(1, OVTK_StimulationId_Label_1E, box:get_current_time(), 0)
    wait_for_id(box, 2)
    
    
    box:log("Info", "experiment start")
    box:send_stimulation(1, OVTK_StimulationId_ExperimentStart, box:get_current_time(), 0)
    
    images_iter = 1
    is_finished = false
    user_score = 0
    response_time = 0.0
    while images_iter <= #sequence do
        
        if images_iter%block_size==0 and images_iter~=#sequence then
            box:send_stimulation(1, OVTK_StimulationId_RestStart, box:get_current_time(), 0)
            wait_for_sec(box, inter_block_rest)
            box:send_stimulation(1, OVTK_StimulationId_RestStop, box:get_current_time(), 0)
            box:send_stimulation(1, OVTK_StimulationId_Label_19, box:get_current_time(), 0)
        end
        
        wait_for_sec(box, rest_duration)
        
        -- display a cross on screen
		box:send_stimulation(1, OVTK_GDF_Start_Of_Trial, box:get_current_time(), 0)
		box:send_stimulation(1, OVTK_GDF_Cross_On_Screen, box:get_current_time(), 0)
		wait_for_sec(box, cross_duration)

		-- clear cross
		box:send_stimulation(1, OVTK_StimulationId_Label_19, box:get_current_time(), 0)
		wait_for_sec(box, post_cross_duration)
             
        -- clear input buffer before face assessment
        for stimulation = 1, box:get_stimulation_count(1) do
            box:remove_stimulation(1, 1)
        end
		
		-- display cue
        display_timestamp = box:get_current_time()
		box:send_stimulation(1, sequence[images_iter], box:get_current_time(), 0)
        
		-- clear cue
        t = box:get_current_time() + display_cue_duration
		box:send_stimulation(1, OVTK_StimulationId_Label_19, t, 0)
        box:send_stimulation(1, OVTK_GDF_End_Of_Trial, t, 0)
        
        is_assesed = false
        while box:keep_processing() do
            -- check keyboard input for stimulations
            for stimulation = 1, box:get_stimulation_count(1) do
                -- gets the received stimulation
                id, timestamp, duration = box:get_stimulation(1, 1)
                -- remove stimulation from buffer
                box:remove_stimulation(1, 1)
                -- even ids mean key releases (for ids look at keyboard_to_stimulation.txt)
                box:log("Info", string.format("At time %f on input %i got stimulation id:%s date:%s duration:%s", box:get_current_time(), 1, id, timestamp, duration))            
                if id%2 == 0 then
                    is_assesed = true                   
                
                    if id == 6 then
                        is_finished = true
                        break
                    end
                    
                    is_good = is_answer_good(task_FET, task_Z, sequence[images_iter], id)
                    
                    if is_good then
                        box:send_stimulation(1, OVTK_GDF_Right, box:get_current_time(), 0) 
                        user_score = user_score + 1
                    else
                        box:send_stimulation(1, OVTK_GDF_Left, box:get_current_time(), 0)
                    end
                else
                    response_time = response_time + box:get_current_time() - display_timestamp                  
                end
              
            end
            
            box:sleep()
            
            if is_assesed or is_finished then break end
        
        end
             
        if is_finished then break end
        
        images_iter = images_iter + 1
    end
    
    images_iter = images_iter - 1
    box:log("Info", string.format("User score: %i / %i (%.2f%%), avg. time: %.4f", user_score, images_iter, 100 * user_score / images_iter, response_time / images_iter))
    
	box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, box:get_current_time() + rest_duration, 0)
end
