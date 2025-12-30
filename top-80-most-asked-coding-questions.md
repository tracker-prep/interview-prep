
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

## 31. Spiral Matrix

**Problem:** Return all elements of a matrix in spiral order.
🔗 [https://leetcode.com/problems/spiral-matrix/](https://leetcode.com/problems/spiral-matrix/)

**Idea:** Peel layers one by one.

```python
def spiralOrder(matrix):
    res = []
    while matrix:
        res += matrix.pop(0)
        matrix = list(zip(*matrix))[::-1]
    return res
```

**Time:** O(mn)
**Space:** O(1) extra

---

## 32. Subsets

**Problem:** Return all possible subsets (power set).
🔗 [https://leetcode.com/problems/subsets/](https://leetcode.com/problems/subsets/)

**Idea:** Backtracking – choose or skip.

```python
def subsets(nums):
    res = []
    def dfs(i, path):
        res.append(path)
        for j in range(i, len(nums)):
            dfs(j+1, path+[nums[j]])
    dfs(0, [])
    return res
```

**Time:** O(2ⁿ)
**Space:** O(n)

---

## 33. Permutations

**Problem:** Return all permutations of numbers.
🔗 [https://leetcode.com/problems/permutations/](https://leetcode.com/problems/permutations/)

```python
def permute(nums):
    res = []
    def dfs(path, used):
        if len(path) == len(nums):
            res.append(path)
            return
        for i in range(len(nums)):
            if i in used: 
                continue
            dfs(path+[nums[i]], used|{i})
    dfs([], set())
    return res
```

**Time:** O(n · n!)
**Space:** O(n)

---

## 34. Kth Largest Element in an Array

**Problem:** Find kth largest element.
🔗 [https://leetcode.com/problems/kth-largest-element-in-an-array/](https://leetcode.com/problems/kth-largest-element-in-an-array/)

```python
import heapq

def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]
```

**Time:** O(n log k)
**Space:** O(k)

---

## 35. Top K Frequent Elements

**Problem:** Return k most frequent elements.
🔗 [https://leetcode.com/problems/top-k-frequent-elements/](https://leetcode.com/problems/top-k-frequent-elements/)

```python
from collections import Counter
import heapq

def topKFrequent(nums, k):
    cnt = Counter(nums)
    return [x for x,_ in heapq.nlargest(k, cnt.items(), key=lambda x: x[1])]
```

**Time:** O(n log k)
**Space:** O(n)

---

## 36. Binary Tree Right Side View

**Problem:** Return values visible from right side.
🔗 [https://leetcode.com/problems/binary-tree-right-side-view/](https://leetcode.com/problems/binary-tree-right-side-view/)

```python
def rightSideView(root):
    res = []
    def dfs(node, depth):
        if not node:
            return
        if depth == len(res):
            res.append(node.val)
        dfs(node.right, depth+1)
        dfs(node.left, depth+1)
    dfs(root, 0)
    return res
```

**Time:** O(n)
**Space:** O(h)

---

## 37. Serialize and Deserialize Binary Tree

**Problem:** Serialize tree to string and back.
🔗 [https://leetcode.com/problems/serialize-and-deserialize-binary-tree/](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

```python
def serialize(root):
    res = []
    def dfs(node):
        if not node:
            res.append('#')
            return
        res.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(res)

def deserialize(data):
    vals = iter(data.split(","))
    def dfs():
        v = next(vals)
        if v == '#':
            return None
        node = TreeNode(int(v))
        node.left = dfs()
        node.right = dfs()
        return node
    return dfs()
```

**Time:** O(n)
**Space:** O(n)

---

## 38. Maximum Depth of Binary Tree

**Problem:** Find max depth of tree.
🔗 [https://leetcode.com/problems/maximum-depth-of-binary-tree/](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

**Time:** O(n)
**Space:** O(h)

---

## 39. Symmetric Tree

**Problem:** Check if tree is symmetric.
🔗 [https://leetcode.com/problems/symmetric-tree/](https://leetcode.com/problems/symmetric-tree/)

```python
def isSymmetric(root):
    def mirror(a, b):
        if not a and not b:
            return True
        if not a or not b:
            return False
        return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)
    return mirror(root, root)
```

**Time:** O(n)
**Space:** O(h)

---

## 40. Flatten Binary Tree to Linked List

**Problem:** Flatten tree to linked list in-place.
🔗 [https://leetcode.com/problems/flatten-binary-tree-to-linked-list/](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

```python
def flatten(root):
    if not root:
        return
    flatten(root.left)
    flatten(root.right)
    tmp = root.right
    root.right = root.left
    root.left = None
    cur = root
    while cur.right:
        cur = cur.right
    cur.right = tmp
```

**Time:** O(n)
**Space:** O(h)

---

## 41. Trapping Rain Water

**Problem:** Compute trapped rainwater.
🔗 [https://leetcode.com/problems/trapping-rain-water/](https://leetcode.com/problems/trapping-rain-water/)

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

**Time:** O(n)
**Space:** O(1)

---

## 42. Minimum Window Substring

**Problem:** Smallest substring containing all characters.
🔗 [https://leetcode.com/problems/minimum-window-substring/](https://leetcode.com/problems/minimum-window-substring/)

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

**Time:** O(n)
**Space:** O(k)

---

## 43. Sliding Window Maximum

**Problem:** Return max in each window of size k.
🔗 [https://leetcode.com/problems/sliding-window-maximum/](https://leetcode.com/problems/sliding-window-maximum/)

```python
from collections import deque

def maxSlidingWindow(nums, k):
    dq, res = deque(), []
    for i,n in enumerate(nums):
        while dq and nums[dq[-1]] < n:
            dq.pop()
        dq.append(i)
        if dq[0] == i-k:
            dq.popleft()
        if i >= k-1:
            res.append(nums[dq[0]])
    return res
```

**Time:** O(n)
**Space:** O(k)

---

## 44. LRU Cache

**Problem:** Design LRU cache.
🔗 [https://leetcode.com/problems/lru-cache/](https://leetcode.com/problems/lru-cache/)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.cap = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

**Time:** O(1)
**Space:** O(capacity)

---

## 45. Find Median from Data Stream

**Problem:** Maintain median of stream.
🔗 [https://leetcode.com/problems/find-median-from-data-stream/](https://leetcode.com/problems/find-median-from-data-stream/)

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap
        self.large = []  # min heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

**Time:** O(log n)
**Space:** O(n)

---

## 46. Kth Largest Element in an Array

**Problem:** Return kth largest element.
🔗 [https://leetcode.com/problems/kth-largest-element-in-an-array/](https://leetcode.com/problems/kth-largest-element-in-an-array/)

```python
import heapq

def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]
```

**Time:** O(n log k)
**Space:** O(k)

---

## 47. Set Matrix Zeroes

**Problem:** Set rows & columns to zero.
🔗 [https://leetcode.com/problems/set-matrix-zeroes/](https://leetcode.com/problems/set-matrix-zeroes/)

```python
def setZeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    row0 = any(matrix[0][j] == 0 for j in range(cols))
    col0 = any(matrix[i][0] == 0 for i in range(rows))

    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0

    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    if row0:
        for j in range(cols): matrix[0][j] = 0
    if col0:
        for i in range(rows): matrix[i][0] = 0
```

**Time:** O(mn)
**Space:** O(1)

---

## 48. Rotate Array

**Problem:** Rotate array by k steps.
🔗 [https://leetcode.com/problems/rotate-array/](https://leetcode.com/problems/rotate-array/)

```python
def rotate(nums, k):
    k %= len(nums)
    nums[:] = nums[-k:] + nums[:-k]
```

**Time:** O(n)
**Space:** O(1)

---

## 49. Binary Tree Right Side View

**Problem:** Right view of tree.
🔗 [https://leetcode.com/problems/binary-tree-right-side-view/](https://leetcode.com/problems/binary-tree-right-side-view/)

```python
def rightSideView(root):
    res = []
    def dfs(node, depth):
        if not node:
            return
        if depth == len(res):
            res.append(node.val)
        dfs(node.right, depth+1)
        dfs(node.left, depth+1)
    dfs(root, 0)
    return res
```

**Time:** O(n)
**Space:** O(h)

---

## 50. Serialize and Deserialize Binary Tree

**Problem:** Serialize and deserialize binary tree.
🔗 [https://leetcode.com/problems/serialize-and-deserialize-binary-tree/](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

```python
def serialize(root):
    res = []
    def dfs(node):
        if not node:
            res.append('#')
            return
        res.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(res)
```

**Time:** O(n)
**Space:** O(n)

---

## 51. Maximum Depth of Binary Tree

**Problem:** Return the maximum depth of a binary tree.
🔗 [https://leetcode.com/problems/maximum-depth-of-binary-tree/](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

**Time:** O(n)
**Space:** O(h)

---

## 52. Symmetric Tree

**Problem:** Check whether a tree is symmetric around its center.
🔗 [https://leetcode.com/problems/symmetric-tree/](https://leetcode.com/problems/symmetric-tree/)

```python
def isSymmetric(root):
    def mirror(a, b):
        if not a and not b:
            return True
        if not a or not b:
            return False
        return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)
    return mirror(root, root)
```

**Time:** O(n)
**Space:** O(h)

---

## 53. Flatten Binary Tree to Linked List

**Problem:** Flatten a binary tree to a linked list in-place.
🔗 [https://leetcode.com/problems/flatten-binary-tree-to-linked-list/](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

```python
def flatten(root):
    if not root:
        return
    flatten(root.left)
    flatten(root.right)
    tmp = root.right
    root.right = root.left
    root.left = None
    cur = root
    while cur.right:
        cur = cur.right
    cur.right = tmp
```

**Time:** O(n)
**Space:** O(h)

---

## 54. Find Peak Element

**Problem:** Find a peak element and return its index.
🔗 [https://leetcode.com/problems/find-peak-element/](https://leetcode.com/problems/find-peak-element/)

```python
def findPeakElement(nums):
    l, r = 0, len(nums)-1
    while l < r:
        m = (l + r) // 2
        if nums[m] > nums[m+1]:
            r = m
        else:
            l = m + 1
    return l
```

**Time:** O(log n)
**Space:** O(1)

---

## 55. Subsets

**Problem:** Return all possible subsets of a set.
🔗 [https://leetcode.com/problems/subsets/](https://leetcode.com/problems/subsets/)

```python
def subsets(nums):
    res = []
    def dfs(i, path):
        res.append(path)
        for j in range(i, len(nums)):
            dfs(j+1, path+[nums[j]])
    dfs(0, [])
    return res
```

**Time:** O(2ⁿ)
**Space:** O(n)

---

## 56. Permutations

**Problem:** Return all permutations of numbers.
🔗 [https://leetcode.com/problems/permutations/](https://leetcode.com/problems/permutations/)

```python
def permute(nums):
    res = []
    def dfs(path, used):
        if len(path) == len(nums):
            res.append(path)
            return
        for i in range(len(nums)):
            if i in used:
                continue
            dfs(path+[nums[i]], used|{i})
    dfs([], set())
    return res
```

**Time:** O(n · n!)
**Space:** O(n)

---

## 57. Combination Sum II

**Problem:** Find unique combinations that sum to target (each number once).
🔗 [https://leetcode.com/problems/combination-sum-ii/](https://leetcode.com/problems/combination-sum-ii/)

```python
def combinationSum2(candidates, target):
    candidates.sort()
    res = []
    def dfs(start, path, total):
        if total == target:
            res.append(path)
            return
        if total > target:
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i-1]:
                continue
            dfs(i+1, path+[candidates[i]], total+candidates[i])
    dfs(0, [], 0)
    return res
```

**Time:** Exponential
**Space:** O(n)

---

## 58. Gas Station

**Problem:** Find starting gas station to complete circuit.
🔗 [https://leetcode.com/problems/gas-station/](https://leetcode.com/problems/gas-station/)

```python
def canCompleteCircuit(gas, cost):
    total = tank = start = 0
    for i in range(len(gas)):
        total += gas[i] - cost[i]
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start if total >= 0 else -1
```

**Time:** O(n)
**Space:** O(1)

---

## 59. Valid Sudoku

**Problem:** Determine if a Sudoku board is valid.
🔗 [https://leetcode.com/problems/valid-sudoku/](https://leetcode.com/problems/valid-sudoku/)

```python
def isValidSudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == '.':
                continue
            b = (r//3)*3 + c//3
            if v in rows[r] or v in cols[c] or v in boxes[b]:
                return False
            rows[r].add(v)
            cols[c].add(v)
            boxes[b].add(v)
    return True
```

**Time:** O(1)
**Space:** O(1)

---

## 60. Sudoku Solver

**Problem:** Solve a Sudoku puzzle.
🔗 [https://leetcode.com/problems/sudoku-solver/](https://leetcode.com/problems/sudoku-solver/)

```python
def solveSudoku(board):
    def valid(r, c, v):
        for i in range(9):
            if board[r][i] == v or board[i][c] == v:
                return False
            if board[(r//3)*3+i//3][(c//3)*3+i%3] == v:
                return False
        return True

    def dfs():
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for v in "123456789":
                        if valid(r, c, v):
                            board[r][c] = v
                            if dfs():
                                return True
                            board[r][c] = '.'
                    return False
        return True

    dfs()
```

**Time:** Exponential
**Space:** O(1)

---

## 61. Meeting Rooms II

**Problem:** Minimum number of meeting rooms required.
🔗 [https://leetcode.com/problems/meeting-rooms-ii/](https://leetcode.com/problems/meeting-rooms-ii/)

```python
import heapq

def minMeetingRooms(intervals):
    intervals.sort()
    heap = []
    for s,e in intervals:
        if heap and heap[0] <= s:
            heapq.heappop(heap)
        heapq.heappush(heap, e)
    return len(heap)
```

**Time:** O(n log n)
**Space:** O(n)

---

## 62. Task Scheduler

**Problem:** Minimum intervals to execute tasks with cooldown.
🔗 [https://leetcode.com/problems/task-scheduler/](https://leetcode.com/problems/task-scheduler/)

```python
from collections import Counter

def leastInterval(tasks, n):
    freq = Counter(tasks)
    maxf = max(freq.values())
    cnt = sum(1 for v in freq.values() if v == maxf)
    return max(len(tasks), (maxf-1)*(n+1) + cnt)
```

**Time:** O(n)
**Space:** O(1)

---

## 63. Alien Dictionary

**Problem:** Determine order of characters in alien language.
🔗 [https://leetcode.com/problems/alien-dictionary/](https://leetcode.com/problems/alien-dictionary/)

```python
from collections import defaultdict, deque

def alienOrder(words):
    graph = defaultdict(set)
    indeg = {c:0 for w in words for c in w}

    for w1, w2 in zip(words, words[1:]):
        for a,b in zip(w1, w2):
            if a != b:
                if b not in graph[a]:
                    graph[a].add(b)
                    indeg[b] += 1
                break
        else:
            if len(w1) > len(w2):
                return ""

    q = deque([c for c in indeg if indeg[c] == 0])
    res = []
    while q:
        c = q.popleft()
        res.append(c)
        for nei in graph[c]:
            indeg[nei] -= 1
            if indeg[nei] == 0:
                q.append(nei)
    return "".join(res) if len(res) == len(indeg) else ""
```

**Time:** O(V + E)
**Space:** O(V)

---

## 64. Longest Consecutive Sequence

**Problem:** Longest sequence of consecutive integers.
🔗 [https://leetcode.com/problems/longest-consecutive-sequence/](https://leetcode.com/problems/longest-consecutive-sequence/)

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

**Time:** O(n)
**Space:** O(n)

---

## 65. Edit Distance

**Problem:** Minimum operations to convert one word to another.
🔗 [https://leetcode.com/problems/edit-distance/](https://leetcode.com/problems/edit-distance/)

```python
def minDistance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

**Time:** O(mn)
**Space:** O(mn)

---

## 66. Burst Balloons

**Problem:** Max coins by bursting balloons wisely.
🔗 [https://leetcode.com/problems/burst-balloons/](https://leetcode.com/problems/burst-balloons/)

```python
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n):
        for l in range(n-length):
            r = l + length
            for k in range(l+1, r):
                dp[l][r] = max(dp[l][r],
                               nums[l]*nums[k]*nums[r] + dp[l][k] + dp[k][r])
    return dp[0][-1]
```

**Time:** O(n³)
**Space:** O(n²)

---

## 67. Word Ladder

**Problem:** Shortest transformation sequence length.
🔗 [https://leetcode.com/problems/word-ladder/](https://leetcode.com/problems/word-ladder/)

```python
from collections import deque

def ladderLength(begin, end, wordList):
    wordList = set(wordList)
    q = deque([(begin, 1)])
    while q:
        w, d = q.popleft()
        if w == end:
            return d
        for i in range(len(w)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                nw = w[:i] + c + w[i+1:]
                if nw in wordList:
                    wordList.remove(nw)
                    q.append((nw, d+1))
    return 0
```

**Time:** O(n · 26 · L)
**Space:** O(n)

---

## 68. Palindromic Substrings

**Problem:** Count palindromic substrings.
🔗 [https://leetcode.com/problems/palindromic-substrings/](https://leetcode.com/problems/palindromic-substrings/)

```python
def countSubstrings(s):
    res = 0
    for i in range(len(s)):
        for l, r in [(i,i), (i,i+1)]:
            while l >= 0 and r < len(s) and s[l] == s[r]:
                res += 1
                l -= 1
                r += 1
    return res
```

**Time:** O(n²)
**Space:** O(1)

---

## 69. Find the Duplicate Number

**Problem:** Find duplicate without modifying array.
🔗 [https://leetcode.com/problems/find-the-duplicate-number/](https://leetcode.com/problems/find-the-duplicate-number/)

```python
def findDuplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```

**Time:** O(n)
**Space:** O(1)

---

## 70. Binary Tree Maximum Path Sum

**Problem:** Maximum path sum in tree.
🔗 [https://leetcode.com/problems/binary-tree-maximum-path-sum/](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

```python
def maxPathSum(root):
    res = float('-inf')
    def dfs(node):
        nonlocal res
        if not node:
            return 0
        l = max(dfs(node.left), 0)
        r = max(dfs(node.right), 0)
        res = max(res, node.val + l + r)
        return node.val + max(l, r)
    dfs(root)
    return res
```

**Time:** O(n)
**Space:** O(h)

---

## 71. Minimum Cost to Cut a Stick

**Problem:** Minimum total cost of cuts.
🔗 [https://leetcode.com/problems/minimum-cost-to-cut-a-stick/](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/)

```python
def minCost(n, cuts):
    cuts = [0] + sorted(cuts) + [n]
    m = len(cuts)
    dp = [[0]*m for _ in range(m)]
    for length in range(2, m):
        for i in range(m-length):
            j = i + length
            dp[i][j] = min(
                cuts[j] - cuts[i] + dp[i][k] + dp[k][j]
                for k in range(i+1, j)
            )
    return dp[0][-1]
```

**Time:** O(n³)
**Space:** O(n²)

---

## 72. Regular Expression Matching

**Problem:** Regex matching with `.` and `*`.
🔗 [https://leetcode.com/problems/regular-expression-matching/](https://leetcode.com/problems/regular-expression-matching/)

```python
def isMatch(s, p):
    dp = [[False]*(len(p)+1) for _ in range(len(s)+1)]
    dp[0][0] = True

    for j in range(2, len(p)+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, len(s)+1):
        for j in range(1, len(p)+1):
            if p[j-1] in {s[i-1], '.'}:
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i][j-2] or \
                           (p[j-2] in {s[i-1], '.'} and dp[i-1][j])
    return dp[-1][-1]
```

**Time:** O(mn)
**Space:** O(mn)

---

## 73. Maximal Rectangle

**Problem:** Largest rectangle of 1’s in a binary matrix.
🔗 [https://leetcode.com/problems/maximal-rectangle/](https://leetcode.com/problems/maximal-rectangle/)

```python
def maximalRectangle(matrix):
    if not matrix:
        return 0
    heights = [0]*len(matrix[0])
    res = 0
    for row in matrix:
        for i,v in enumerate(row):
            heights[i] = heights[i]+1 if v=='1' else 0
        stack = []
        for i,h in enumerate(heights+[0]):
            while stack and heights[stack[-1]] > h:
                H = heights[stack.pop()]
                W = i if not stack else i-stack[-1]-1
                res = max(res, H*W)
            stack.append(i)
    return res
```

**Time:** O(mn)
**Space:** O(n)

---

## 74. Largest Rectangle in Histogram

**Problem:** Largest rectangle area in histogram.
🔗 [https://leetcode.com/problems/largest-rectangle-in-histogram/](https://leetcode.com/problems/largest-rectangle-in-histogram/)

```python
def largestRectangleArea(heights):
    stack, res = [], 0
    for i,h in enumerate(heights+[0]):
        while stack and heights[stack[-1]] > h:
            H = heights[stack.pop()]
            W = i if not stack else i-stack[-1]-1
            res = max(res, H*W)
        stack.append(i)
    return res
```

**Time:** O(n)
**Space:** O(n)

---

## 75. Median of Two Sorted Arrays

**Problem:** Median of two sorted arrays.
🔗 [https://leetcode.com/problems/median-of-two-sorted-arrays/](https://leetcode.com/problems/median-of-two-sorted-arrays/)

*(Same solution as Problem 13)*

---

## 76. Count of Smaller Numbers After Self

**Problem:** Count smaller numbers after each element.
🔗 [https://leetcode.com/problems/count-of-smaller-numbers-after-self/](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

```python
def countSmaller(nums):
    res = [0]*len(nums)
    enum = list(enumerate(nums))

    def sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr)//2
        left, right = sort(arr[:mid]), sort(arr[mid:])
        i = 0
        for l in left:
            while i < len(right) and right[i][1] < l[1]:
                i += 1
            res[l[0]] += i
        return sorted(left+right, key=lambda x: x[1])

    sort(enum)
    return res
```

**Time:** O(n log n)
**Space:** O(n)

---

## 77. Course Schedule II

**Problem:** Return course order if possible.
🔗 [https://leetcode.com/problems/course-schedule-ii/](https://leetcode.com/problems/course-schedule-ii/)

```python
from collections import defaultdict, deque

def findOrder(n, prereq):
    g = defaultdict(list)
    indeg = [0]*n
    for a,b in prereq:
        g[b].append(a)
        indeg[a]+=1

    q = deque([i for i in range(n) if indeg[i]==0])
    res=[]
    while q:
        c=q.popleft()
        res.append(c)
        for nei in g[c]:
            indeg[nei]-=1
            if indeg[nei]==0:
                q.append(nei)
    return res if len(res)==n else []
```

**Time:** O(V+E)
**Space:** O(V)

---

## 78. Minimum Number of Refueling Stops

**Problem:** Minimum refuels to reach target.
🔗 [https://leetcode.com/problems/minimum-number-of-refueling-stops/](https://leetcode.com/problems/minimum-number-of-refueling-stops/)

```python
import heapq

def minRefuelStops(target, fuel, stations):
    heap = []
    i = res = 0
    while fuel < target:
        while i < len(stations) and stations[i][0] <= fuel:
            heapq.heappush(heap, -stations[i][1])
            i += 1
        if not heap:
            return -1
        fuel += -heapq.heappop(heap)
        res += 1
    return res
```

**Time:** O(n log n)
**Space:** O(n)

---

## 79. Maximum Product Subarray

**Problem:** Maximum product of a contiguous subarray.
🔗 [https://leetcode.com/problems/maximum-product-subarray/](https://leetcode.com/problems/maximum-product-subarray/)

```python
def maxProduct(nums):
    cur_max = cur_min = res = nums[0]
    for n in nums[1:]:
        tmp = cur_max
        cur_max = max(n, tmp*n, cur_min*n)
        cur_min = min(n, tmp*n, cur_min*n)
        res = max(res, cur_max)
    return res
```

**Time:** O(n)
**Space:** O(1)

---

## 80. Longest Valid Parentheses

**Problem:** Length of longest valid parentheses substring.
🔗 [https://leetcode.com/problems/longest-valid-parentheses/](https://leetcode.com/problems/longest-valid-parentheses/)

```python
def longestValidParentheses(s):
    stack = [-1]
    res = 0
    for i,c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                res = max(res, i-stack[-1])
    return res
```

**Time:** O(n)
**Space:** O(n)

---
