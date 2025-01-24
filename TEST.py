from operator import index

from fontTools.merge.util import first

nums = [3,2,3]
nums = [0,3,-3,4,-1]
nums = [2,1,9,4,4,56,90,3]
nums =  [0,3,-3,4,-1]
target = -1






def twoSum(nums,target):
    filter_nums=nums[:]
    filter_nums.sort()
    while True:
        first_number=filter_nums[0]
        for second_index in range(1,len(filter_nums)):
            second_number=filter_nums[second_index]
            if (first_number+second_number or second_number+first_number)==target:
                if first_number==second_number:
                    index_0=nums.index(first_number)
                    index_1=nums[index_0+1:].index(second_number)+index_0+1
                else:
                    index_0=nums.index(first_number)
                    index_1=nums.index(second_number)
                return [index_0, index_1]
            elif (first_number+second_number)>target:
                filter_nums=filter_nums[:second_index]
                break
        filter_nums=filter_nums[1:]

def isValid(self, s: str) -> bool:
    for x in s:
        if x == "(":
            if ")" in s:

                s[:s.index(")")] = ""


        elif x == "[":
            None

        elif x == "{":
            None
