
# FAANG / Uber / Stripe / Salesforce – Top 80 Coding Interview Problems

This document contains **80 high‑frequency interview problems** used by FAANG and top‑paying companies
(Uber, Stripe, Airbnb, Salesforce, Databricks, etc.).

---

## 1. Two Sum
**Problem:** Given an array of integers, return indices of two numbers that add up to a target.  
🔗 https://leetcode.com/problems/two-sum/

```python
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
```
Time: O(n) | Space: O(n)

---

## 2. Product of Array Except Self
**Problem:** Return array where each element is product of all others.  
🔗 https://leetcode.com/problems/product-of-array-except-self/

```python
def productExceptSelf(nums):
    res = [1]*len(nums)
    p = 1
    for i in range(len(nums)):
        res[i] = p
        p *= nums[i]
    s = 1
    for i in range(len(nums)-1,-1,-1):
        res[i] *= s
        s *= nums[i]
    return res
```
Time: O(n) | Space: O(1)

---

## 3. Longest Consecutive Sequence
**Problem:** Find length of longest consecutive elements sequence.  
🔗 https://leetcode.com/problems/longest-consecutive-sequence/

```python
def longestConsecutive(nums):
    s = set(nums)
    best = 0
    for n in s:
        if n-1 not in s:
            cur, length = n, 1
            while cur+1 in s:
                cur += 1
                length += 1
            best = max(best, length)
    return best
```
Time: O(n) | Space: O(n)

---

## 4. Container With Most Water
**Problem:** Max area formed between two vertical lines.  
🔗 https://leetcode.com/problems/container-with-most-water/

```python
def maxArea(height):
    l, r = 0, len(height)-1
    ans = 0
    while l < r:
        ans = max(ans, min(height[l], height[r]) * (r-l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return ans
```
Time: O(n) | Space: O(1)

---

## 5. Trapping Rain Water
**Problem:** Compute trapped rainwater between bars.  
🔗 https://leetcode.com/problems/trapping-rain-water/

```python
def trap(height):
    l, r = 0, len(height)-1
    lm = rm = 0
    res = 0
    while l < r:
        if height[l] < height[r]:
            lm = max(lm, height[l])
            res += lm - height[l]
            l += 1
        else:
            rm = max(rm, height[r])
            res += rm - height[r]
            r -= 1
    return res
```
Time: O(n) | Space: O(1)

---

## 6. Longest Substring Without Repeating Characters
**Problem:** Length of longest substring with unique characters.  
🔗 https://leetcode.com/problems/longest-substring-without-repeating-characters/

```python
def lengthOfLongestSubstring(s):
    seen, l, res = {}, 0, 0
    for r, c in enumerate(s):
        if c in seen and seen[c] >= l:
            l = seen[c] + 1
        seen[c] = r
        res = max(res, r-l+1)
    return res
```
Time: O(n) | Space: O(n)

---

## 7. Minimum Window Substring
**Problem:** Smallest substring containing all chars of t.  
🔗 https://leetcode.com/problems/minimum-window-substring/

```python
from collections import Counter

def minWindow(s, t):
    need = Counter(t)
    missing = len(t)
    l = start = end = 0
    for r, c in enumerate(s, 1):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        if missing == 0:
            while l < r and need[s[l]] < 0:
                need[s[l]] += 1
                l += 1
            if end == 0 or r-l < end-start:
                start, end = l, r
            need[s[l]] += 1
            missing += 1
            l += 1
    return s[start:end]
```
Time: O(n) | Space: O(k)

---

## 8. Valid Parentheses
**Problem:** Check if parentheses string is valid.  
🔗 https://leetcode.com/problems/valid-parentheses/

```python
def isValid(s):
    stack = []
    mp = {')':'(', ']':'[', '}':'{'}
    for c in s:
        if c in mp:
            if not stack or stack.pop() != mp[c]:
                return False
        else:
            stack.append(c)
    return not stack
```
Time: O(n) | Space: O(n)

---

## 9. Merge Two Sorted Lists
**Problem:** Merge two sorted linked lists.  
🔗 https://leetcode.com/problems/merge-two-sorted-lists/

```python
def mergeTwoLists(l1, l2):
    dummy = cur = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            cur.next, l1 = l1, l1.next
        else:
            cur.next, l2 = l2, l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```
Time: O(n+m) | Space: O(1)

---

## 10. Remove Nth Node From End
**Problem:** Remove nth node from end of linked list.  
🔗 https://leetcode.com/problems/remove-nth-node-from-end-of-list/

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n+1):
        fast = fast.next
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```
Time: O(n) | Space: O(1)

---

## 11. Search in Rotated Sorted Array

**Problem:** Search a target in a rotated sorted array.
🔗 [https://leetcode.com/problems/search-in-rotated-sorted-array/](https://leetcode.com/problems/search-in-rotated-sorted-array/)

**Idea:** Modified binary search using sorted half detection.

```python
def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if nums[m] < target <= nums[r]:
                l = m + 1
            else:
                r = m - 1
    return -1
```

**Time:** O(log n)
**Space:** O(1)

---

## 12. Find Minimum in Rotated Sorted Array

**Problem:** Find the minimum element in a rotated sorted array.
🔗 [https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

**Idea:** Binary search comparing mid with right boundary.

```python
def findMin(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        m = (l + r) // 2
        if nums[m] > nums[r]:
            l = m + 1
        else:
            r = m
    return nums[l]
```

**Time:** O(log n)
**Space:** O(1)

---

## 13. Median of Two Sorted Arrays

**Problem:** Find median of two sorted arrays.
🔗 [https://leetcode.com/problems/median-of-two-sorted-arrays/](https://leetcode.com/problems/median-of-two-sorted-arrays/)

**Idea:** Binary search on partitions.

```python
def findMedianSortedArrays(a, b):
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    l, r = 0, m
    while l <= r:
        i = (l + r) // 2
        j = (m + n + 1) // 2 - i
        L1 = a[i-1] if i else float('-inf')
        R1 = a[i] if i < m else float('inf')
        L2 = b[j-1] if j else float('-inf')
        R2 = b[j] if j < n else float('inf')
        if L1 <= R2 and L2 <= R1:
            if (m + n) % 2:
                return max(L1, L2)
            return (max(L1, L2) + min(R1, R2)) / 2
        elif L1 > R2:
            r = i - 1
        else:
            l = i + 1
```

**Time:** O(log(min(m,n)))
**Space:** O(1)

---

## 14. Binary Tree Level Order Traversal

**Problem:** Return level order traversal of a binary tree.
🔗 [https://leetcode.com/problems/binary-tree-level-order-traversal/](https://leetcode.com/problems/binary-tree-level-order-traversal/)

**Idea:** BFS using queue.

```python
from collections import deque

def levelOrder(root):
    if not root:
        return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

**Time:** O(n)
**Space:** O(n)

---

## 15. Validate Binary Search Tree

**Problem:** Check if a binary tree is a valid BST.
🔗 [https://leetcode.com/problems/validate-binary-search-tree/](https://leetcode.com/problems/validate-binary-search-tree/)

**Idea:** DFS with value bounds.

```python
def isValidBST(root):
    def dfs(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)
    return dfs(root, float('-inf'), float('inf'))
```

**Time:** O(n)
**Space:** O(h)

---

## 16. Diameter of Binary Tree

**Problem:** Return the length of the diameter of the tree.
🔗 [https://leetcode.com/problems/diameter-of-binary-tree/](https://leetcode.com/problems/diameter-of-binary-tree/)

**Idea:** DFS computing height and tracking max path.

```python
def diameterOfBinaryTree(root):
    ans = 0
    def dfs(node):
        nonlocal ans
        if not node:
            return 0
        l = dfs(node.left)
        r = dfs(node.right)
        ans = max(ans, l + r)
        return 1 + max(l, r)
    dfs(root)
    return ans
```

**Time:** O(n)
**Space:** O(h)

---

## 17. Number of Islands

**Problem:** Count number of islands in a grid.
🔗 [https://leetcode.com/problems/number-of-islands/](https://leetcode.com/problems/number-of-islands/)

**Idea:** DFS flood-fill.

```python
def numIslands(grid):
    def dfs(r, c):
        if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        dfs(r+1,c); dfs(r-1,c); dfs(r,c+1); dfs(r,c-1)

    count = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '1':
                dfs(r,c)
                count += 1
    return count
```

**Time:** O(mn)
**Space:** O(mn)

---

## 18. Clone Graph

**Problem:** Deep copy an undirected graph.
🔗 [https://leetcode.com/problems/clone-graph/](https://leetcode.com/problems/clone-graph/)

**Idea:** DFS + hashmap.

```python
def cloneGraph(node):
    if not node:
        return None
    mp = {}
    def dfs(n):
        if n in mp:
            return mp[n]
        copy = Node(n.val)
        mp[n] = copy
        for nei in n.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node)
```

**Time:** O(V + E)
**Space:** O(V)

---

## 19. Course Schedule

**Problem:** Determine if all courses can be finished.
🔗 [https://leetcode.com/problems/course-schedule/](https://leetcode.com/problems/course-schedule/)

**Idea:** Cycle detection using DFS.

```python
from collections import defaultdict

def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    for a,b in prerequisites:
        graph[a].append(b)
    visiting, visited = set(), set()

    def dfs(c):
        if c in visiting:
            return False
        if c in visited:
            return True
        visiting.add(c)
        for p in graph[c]:
            if not dfs(p):
                return False
        visiting.remove(c)
        visited.add(c)
        return True

    return all(dfs(i) for i in range(numCourses))
```

**Time:** O(V + E)
**Space:** O(V)

---

## 20. Word Search

**Problem:** Check if a word exists in a grid.
🔗 [https://leetcode.com/problems/word-search/](https://leetcode.com/problems/word-search/)

**Idea:** Backtracking DFS.

```python
def exist(board, word):
    R, C = len(board), len(board[0])

    def dfs(r, c, i):
        if i == len(word):
            return True
        if r<0 or c<0 or r>=R or c>=C or board[r][c] != word[i]:
            return False
        tmp = board[r][c]
        board[r][c] = '#'
        found = dfs(r+1,c,i+1) or dfs(r-1,c,i+1) or dfs(r,c+1,i+1) or dfs(r,c-1,i+1)
        board[r][c] = tmp
        return found

    for r in range(R):
        for c in range(C):
            if dfs(r,c,0):
                return True
    return False
```

**Time:** O(mn · 4ᵏ)
**Space:** O(k)

---

## 21. Word Break

**Problem:** Determine if string can be segmented into dictionary words.
🔗 [https://leetcode.com/problems/word-break/](https://leetcode.com/problems/word-break/)

```python
def wordBreak(s, wordDict):
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for w in wordDict:
            if i >= len(w) and dp[i-len(w)] and s[i-len(w):i] == w:
                dp[i] = True
    return dp[-1]
```

**Time:** O(n·m·k)
**Space:** O(n)

---

## 22. Decode Ways

**Problem:** Count number of ways to decode string.
🔗 [https://leetcode.com/problems/decode-ways/](https://leetcode.com/problems/decode-ways/)

```python
def numDecodings(s):
    if not s or s[0] == '0':
        return 0
    dp = [0] * (len(s) + 1)
    dp[0] = dp[1] = 1
    for i in range(2, len(s)+1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        if 10 <= int(s[i-2:i]) <= 26:
            dp[i] += dp[i-2]
    return dp[-1]
```

**Time:** O(n)
**Space:** O(n)

---

## 23. Coin Change

**Problem:** Minimum coins to make amount.
🔗 [https://leetcode.com/problems/coin-change/](https://leetcode.com/problems/coin-change/)

```python
def coinChange(coins, amount):
    dp = [amount+1] * (amount+1)
    dp[0] = 0
    for i in range(1, amount+1):
        for c in coins:
            if i >= c:
                dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[amount] if dp[amount] != amount+1 else -1
```

**Time:** O(n·amount)
**Space:** O(amount)

---

## 24. House Robber

**Problem:** Max money without robbing adjacent houses.
🔗 [https://leetcode.com/problems/house-robber/](https://leetcode.com/problems/house-robber/)

```python
def rob(nums):
    prev1 = prev2 = 0
    for n in nums:
        prev1, prev2 = max(prev2+n, prev1), prev1
    return prev1
```

**Time:** O(n)
**Space:** O(1)

---

## 25. Jump Game

**Problem:** Determine if you can reach last index.
🔗 [https://leetcode.com/problems/jump-game/](https://leetcode.com/problems/jump-game/)

```python
def canJump(nums):
    reach = 0
    for i, n in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + n)
    return True
```

**Time:** O(n)
**Space:** O(1)

---

## 26. Merge Intervals

**Problem:** Merge overlapping intervals.
🔗 [https://leetcode.com/problems/merge-intervals/](https://leetcode.com/problems/merge-intervals/)

```python
def merge(intervals):
    intervals.sort()
    res = []
    for s,e in intervals:
        if not res or res[-1][1] < s:
            res.append([s,e])
        else:
            res[-1][1] = max(res[-1][1], e)
    return res
```

**Time:** O(n log n)
**Space:** O(n)

---

## 27. Insert Interval

**Problem:** Insert and merge a new interval.
🔗 [https://leetcode.com/problems/insert-interval/](https://leetcode.com/problems/insert-interval/)

```python
def insert(intervals, newInterval):
    res = []
    for i in intervals:
        if i[1] < newInterval[0]:
            res.append(i)
        elif i[0] > newInterval[1]:
            res.append(newInterval)
            newInterval = i
        else:
            newInterval = [min(i[0], newInterval[0]), max(i[1], newInterval[1])]
    res.append(newInterval)
    return res
```

**Time:** O(n)
**Space:** O(n)

---

## 28. Unique Paths

**Problem:** Count paths in grid from top-left to bottom-right.
🔗 [https://leetcode.com/problems/unique-paths/](https://leetcode.com/problems/unique-paths/)

```python
def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]
```

**Time:** O(mn)
**Space:** O(mn)

---

## 29. Climbing Stairs

**Problem:** Number of distinct ways to climb stairs.
🔗 [https://leetcode.com/problems/climbing-stairs/](https://leetcode.com/problems/climbing-stairs/)

```python
def climbStairs(n):
    a, b = 1, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
```

**Time:** O(n)
**Space:** O(1)

---

## 30. Group Anagrams

**Problem:** Group strings that are anagrams.
🔗 [https://leetcode.com/problems/group-anagrams/](https://leetcode.com/problems/group-anagrams/)

```python
from collections import defaultdict

def groupAnagrams(strs):
    d = defaultdict(list)
    for s in strs:
        d[tuple(sorted(s))].append(s)
    return list(d.values())
```

**Time:** O(n·k log k)
**Space:** O(n·k)

---
