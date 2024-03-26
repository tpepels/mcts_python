import os
import itertools


def compare_files(file1, file2):
    """Compares two files line by line and returns the similarity percentage."""
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        if len(lines1) == 0 or len(lines2) == 0:
            return 0
        total_lines = max(len(lines1), len(lines2))

        # Count the number of similar lines
        common_lines = sum(1 for l1, l2 in zip(lines1, lines2) if l1 == l2)

        return (common_lines / total_lines) * 100


def find_similar_files(directory, min_similarity):
    """Finds and prints pairs of files in the directory that are more than 70% similar."""
    log_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".log")]
    print(f"Found {len(log_files)} log files.")
    max_similarity, total_similarity, n_combinations = 0, 0, 0
    for file1, file2 in itertools.combinations(log_files, 2):
        similarity = compare_files(file1, file2)

        if similarity > max_similarity:
            max_similarity = similarity

        total_similarity += similarity
        n_combinations += 1

        if similarity > min_similarity:
            print(
                f"Files '{os.path.basename(file1)}' and '{os.path.basename(file2)}' are {similarity:.2f}% similar."
            )

    print(f"{max_similarity=:,.2f}")
    print(f"Average similarity: {total_similarity / n_combinations:,.2f}%")


if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    min_similarity = input("Minimum level of similarity (in %): ")
    find_similar_files(directory, float(min_similarity))
