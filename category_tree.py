import os
#define the category tree dictionary
category_tree={}
category_label={}
category=['fruit','device','vegetable','container']
category_tree['fruit']=['lemon','lime','orange','apple','banana','peach','pear']
category_tree['device']=['dry_battery','calculator','stapler','flashlight','lightbulb','keyboard']
category_tree['vegetable']=['potato','bell_pepper','tomato','greens','mushroom','onion']
category_tree['container']=['cereal_box','soda_can','bowl','plate','coffee_mug','water_bottle','pitcher']

category_label['fruit']=[1,0,0,0]
category_label['device']=[0,1,0,0]
category_label['vegetable']=[0,0,1,0]
category_label['container']=[0,0,0,1]
#print(category_tree)
#check if we have the object files in all the category
file_path='/usa/psu/Downloads/rgbd-dataset/'
#print(os.listdir(folder_path))
for c in category:
    for ct_i in category_tree[c]:
        if ct_i not in os.listdir(file_path):
            print("Canot find the object: ",ct_i)



