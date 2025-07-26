import { simpleGit, SimpleGit, SimpleGitOptions } from "simple-git";


/**
 * Clones a Git repository to a specified local path.
 * @param repositoryUrl The URL of the repository to clone.
 * @param localPath The local path where the repository should be cloned.
 * @param options Optional SimpleGit options.
 * @returns A promise that resolves to the SimpleGit instance for the cloned repository.
 */
export async function cloneRepository(
    repositoryUrl: string,
    localPath: string,
    options?: SimpleGitOptions
): Promise<SimpleGit> {
    const git: SimpleGit = simpleGit(options);
    await git.clone(repositoryUrl, localPath);

    return git;
};